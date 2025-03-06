import os
import argparse
import time
import logging
import webbrowser
import gradio
import librosa
import torch
import commons
import utils
import numpy

from torch import LongTensor, no_grad
from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from text.symbols import symbols

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.getLogger('numba').setLevel(logging.WARNING)
limitation = os.getenv("System") == "spaces"        # 限制huggingface空间中的我文本和音频长度

language_marks = {
    "简体中文": "[ZH]",
    "日本語": "[JA]",
    "English": "[EN]",
    "混合": "",
}
lang = ['简体中文', '日本語', 'English', '混合']

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

# 和前端的对应关系
# text=textobx
# language=language_dropdown
# spearker_id=char_dropdown
# noise_scale=noise_scale
# noise_scale_w=noise_scale_w
# speed=duration_slider
#is_symbol=symbol_input的value值
def vits(text, language, speaker_id, noise_scale, noise_scale_w, speed, is_symbol):
    # print(text, language, speaker_id, noise_scale, noise_scale_w, speed, is_symbol)

    start = time.perf_counter()

    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if len(text) > 100 and limitation:
        return f"输入文字过长！{len(text)}>100", None, None

    if language is not None:
        text = language_marks[language] + text + language_marks[language]

    stn_tst = get_text(text, hps, is_symbol)

    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()

    del stn_tst, x_tst, x_tst_lengths, sid

    return "生成成功！", (hps.data.sampling_rate, audio), f"生成耗时 {round(time.perf_counter() - start, 2)} s"

# 和前端的对应关系
# original_speaker=source_speaker
# target_speaker=target_speaker
# record_audio=record_audio
# upload_audio
def conversion(original_speaker, target_speaker, record_audio, upload_audio):
    print(original_speaker, target_speaker, record_audio, upload_audio)
    print(type(original_speaker), type(target_speaker))

    input_audio = record_audio if record_audio is not None else upload_audio
    if input_audio is None:
        return "你需要录制或者上传音频", None
    sampling_rate, audio = input_audio
    # original_speaker_id = original_speaker
    # target_speaker_id = target_speaker

    audio = (audio / numpy.iinfo(audio.dtype).max).astype(numpy.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != hps.data.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
    with no_grad():
        y = torch.FloatTensor(audio)
        y = y / max(-y.min(), y.max()) / 0.99
        y = y.to(device)
        y = y.unsqueeze(0)
        spec = spectrogram_torch(y, hps.data.filter_length,
                                 hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                 center=False).to(device)
        spec_lengths = LongTensor([spec.size(-1)]).to(device)
        sid_src = LongTensor([original_speaker]).to(device)
        sid_tgt = LongTensor([target_speaker]).to(device)
        print(sid_src,sid_tgt)
        audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
            0, 0].data.cpu().float().numpy()

    del y, spec, spec_lengths, sid_src, sid_tgt

    return "转换成功！", (hps.data.sampling_rate, audio)

# 搜索角色
def search_speaker(search_value):
    for s in speakers:
        if search_value == s:
            return s
    for s in speakers:
        if search_value in s:
            return s

# 音素(Phoneme)
def symbol_in(symbol_input, input_text, temp_text):
    return (_clean_text(input_text, hps.data.text_cleaners), input_text) if symbol_input \
        else (temp_text, temp_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./models/G_trilingual.pth", help="模型")
    parser.add_argument("--config_dir", default="./configs/uma_trilingual.json", help="配置文件")
    parser.add_argument('--api', action="store_true", default=False, help="开放后端的REST路由，允许直接向这些端点发出请求，跳过队列。")
    parser.add_argument("--share", action="store_true", default=False, help="公开链接（用于 colab）")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = model.eval()

    _ = utils.load_checkpoint(args.model_dir, model, None)

    # speakers = hps.speakers
    speakers = list(hps.speakers.keys())

    with gradio.Blocks(theme='NoCrypt/miku', css_paths="./themes/theme.css") as app:

        # gradio.Markdown(
        # """
        #     # <center> VITS语音在线合成demo<br>
        #     # <center> 严禁将模型用于任何商业项目，否则后果自负<br>
        #     <div align="center">主要有赛马娘，原神中文，原神日语，崩坏3的音色</div>
        #     <div align="center"><a style="color: red;">结果有随机性，语调可能很奇怪，可多次生成取最佳效果</a></div>
        #     <div align="center"><a style="color: red;">标点符号会影响生成的结果</a></div>
        # """
        # )
        with gradio.Row():
            with gradio.Column(scale=13):
                gradio.Markdown("# VITS语音在线合成demo", elem_classes="center")
                gradio.Markdown("# 严禁将模型用于任何商业项目，否则后果自负", elem_classes="center")
                gradio.Markdown("<div>主要有赛马娘，原神中文，原神日语，崩坏3的音色</div>", elem_classes="center")
                gradio.Markdown("<div><a>结果有随机性，语调可能很奇怪，可多次生成取最佳效果</a></div>", elem_classes="center")
                gradio.Markdown("<div><a>标点符号会影响生成的结果</a></div>", elem_classes="center")
            with gradio.Column(scale=1, min_width=100, elem_classes="dark-light"):
                toggle_dark = gradio.Button(value="切换主题")
                toggle_dark.click(
                    None,
                    js="""
                        () => {
                            document.body.classList.toggle('dark');
                        }
                    """
                )
        with gradio.Tabs():
            with gradio.TabItem("vits"):
                with gradio.Row():
                    with gradio.Column():
                        # 文字输入
                        textbox = gradio.TextArea(label="Text（100字限制）",
                                                 value="今天晚上吃啥好呢。", elem_id=f"input_text")
                        # 选择语言
                        language_dropdown = gradio.Dropdown(choices=lang, value=lang[0], label="Language", type="value")
                        # 音素(Phoneme)输入
                        with gradio.Accordion(label="音素(Phoneme)输入", open=False):
                            temp_text_var = gradio.State()
                            symbol_input = gradio.Checkbox(value=False, label="符号输入")
                            symbol_list = gradio.Dataset(label="符号列表", components=[textbox],
                                                         samples=[[x] for x in symbols], elem_id=f"symbol-list")
                            symbol_list_json = gradio.Json(value=symbols, visible=False)
                        symbol_input.change(symbol_in,
                                            [symbol_input, textbox, temp_text_var],
                                            [textbox, temp_text_var])
                        symbol_list.click(None, [symbol_list, symbol_list_json], textbox,
                                          js="""
                                    (i, symbols, text) => {{
                                        let root = document.querySelector("body > gradio-app");
                                        if (root.shadowRoot != null)
                                            root = root.shadowRoot;
                                        let text_input = root.querySelector("#input_text").querySelector("textarea");
                                        let startPos = text_input.selectionStart;
                                        let endPos = text_input.selectionEnd;
                                        let oldTxt = text_input.value;
                                        let result = oldTxt.substring(0, startPos) + symbols[i] + oldTxt.substring(endPos);
                                        text_input.value = result;
                                        let x = window.scrollX, y = window.scrollY;
                                        text_input.focus();
                                        text_input.selectionStart = startPos + symbols[i].length;
                                        text_input.selectionEnd = startPos + symbols[i].length;
                                        text_input.blur();
                                        window.scrollTo(x, y);
                                        text = text_input.value;
                                        return text;
                                    }}
                                    """
                        )
                        # 搜索角色
                        with gradio.Row():
                            search = gradio.Textbox(label="搜索角色", lines=1)
                            btn2 = gradio.Button(value="搜索")
                        # 选择角色
                        char_dropdown = gradio.Dropdown(choices=speakers, value=speakers[0], label="角色", type="index")
                        # 语音调节
                        with gradio.Row():
                            noise_scale = gradio.Slider(label="noise_scale(控制感情变化程度)", minimum=0.1, maximum=1.0, step=0.1, value=0.667)
                            noise_scale_w = gradio.Slider(label="noise_scale_w(控制音素发音长度)", minimum=0.1, maximum=1.0, step=0.1, value=0.8)
                            duration_slider = gradio.Slider(label="duration_slider(控制整体语速)", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                    with gradio.Column():
                        text_output = gradio.Textbox(label="消息")
                        audio_output = gradio.Audio(label="音频输出", elem_id="tts-audio")
                        time_output = gradio.Textbox(label="额外信息")
                        # 提交
                        btn = gradio.Button(value="生成")

                    btn.click(vits, inputs=[textbox,
                                            language_dropdown,
                                            char_dropdown,
                                            noise_scale,
                                            noise_scale_w,
                                            duration_slider,
                                            symbol_input],
                                    outputs=[text_output, audio_output, time_output])
                    btn2.click(search_speaker, inputs=search, outputs=[char_dropdown])

            with gradio.TabItem("语音转换"):
                gradio.Markdown("""
                                    录制或上传声音，并选择要转换的音色。
                """)
                with gradio.Column():
                    record_audio = gradio.Audio(label="录制您的声音", sources="microphone")
                    upload_audio = gradio.Audio(label="或在此处上传音频", sources="upload")
                    source_speaker = gradio.Dropdown(choices=speakers, value=speakers[0], label="初始角色声音", type="index")
                    target_speaker = gradio.Dropdown(choices=speakers, value=speakers[0], label="目标角色声音", type="index")
                with gradio.Column():
                    message_box = gradio.Textbox(label="消息")
                    converted_audio = gradio.Audio(label='转换的音频')
                btn = gradio.Button("转换!")
                btn.click(conversion, inputs=[source_speaker, target_speaker, record_audio, upload_audio],
                        outputs=[message_box, converted_audio])

            with gradio.TabItem("可用人物预览"):
                gradio.Radio(label="角色", choices=speakers, interactive=False, type="index")

    # 自动打开浏览器
    webbrowser.open("http://127.0.0.1:7860/?__theme=light")
    app.queue(default_concurrency_limit=3, api_open=args.api).launch(share=args.share)
