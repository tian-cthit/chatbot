from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from TTS.api import TTS
from pynput import keyboard
import whisper
import pyaudio
import wave
import torch
import sounddevice as sd


class Recorder:
    def __init__(self,
                 chunksize=8192,
                 dataformat=pyaudio.paInt16,
                 channels=2,
                 rate=44100):
        self.stream = None
        self.wf = None
        self.filename = "output.wav"
        self.chunksize = chunksize
        self.dataformat = dataformat
        self.channels = channels
        self.rate = rate
        self.recording = False
        self.pa = pyaudio.PyAudio()

    def start(self):
        if not self.recording:
            self.wf = wave.open(self.filename, 'wb')
            self.wf.setnchannels(self.channels)
            self.wf.setsampwidth(self.pa.get_sample_size(self.dataformat))
            self.wf.setframerate(self.rate)

            def callback(in_data, frame_count, time_info, status):
                # file write should be able to keep up with audio data stream (about 1378 Kbps)
                self.wf.writeframes(in_data)
                return in_data, pyaudio.paContinue

            self.stream = self.pa.open(format=self.dataformat,
                                       channels=self.channels,
                                       rate=self.rate,
                                       input=True,
                                       stream_callback=callback)
            self.stream.start_stream()
            self.recording = True
            print('Recording started')

    def stop(self):
        if self.recording:
            self.stream.stop_stream()
            self.stream.close()
            self.wf.close()

            self.recording = False
            print('Recording finished')


class Whisper:
    def __init__(self):
        self.model = whisper.load_model("base")

    def audio_to_text(self, audio_file):
        result = self.model.transcribe(audio_file)
        print(result["text"])
        return result["text"]


class BigModel:
    def __init__(self):
        self.model_name = "facebook/blenderbot-400M-distill"
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name).to("cuda:0")
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)

    def blender_process(self, text):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer([text], return_tensors="pt").to(device)
        reply_ids = self.model.generate(**inputs)
        output = self.tokenizer.batch_decode(reply_ids)

        reply_string = output[0].replace("<s>", "")
        reply_string = reply_string.replace("</s>", "")
        reply_string = reply_string.replace("\"", "")
        return reply_string


class TextToSpeech:
    def __init__(self):
        self.model_name = TTS.list_models()[0]
        self.tts = TTS(self.model_name)

    def to_speech(self, text):
        wav = self.tts.tts(text, speaker=self.tts.speakers[0], language=self.tts.languages[0])
        return wav


class KeyboardListener(keyboard.Listener):
    def __init__(self, recorder, audio_text_converter, big_model, text_audio_converter):
        super().__init__(on_press=self.on_press, on_release=self.on_release)
        self.recorder = recorder
        self.audio_text_converter = audio_text_converter
        self.big_model = big_model
        self.text_audio_converter = text_audio_converter

    def on_press(self, key):
        if key == keyboard.Key.enter:
            self.recorder.start()

    def on_release(self, key):
        if key == keyboard.Key.enter:
            self.recorder.stop()
            text = self.audio_text_converter.audio_to_text(self.recorder.filename)
            reply = self.big_model.blender_process(text)
            speech_nparray = self.text_audio_converter.to_speech(reply)
            play_audio(speech_nparray, 18000)
        if key == keyboard.KeyCode.from_char('q'):
            return False


def play_audio(audio_nparray, sps):
    sd.play(audio_nparray, sps)


def main():
    recorder = Recorder()
    audio_text_converter = Whisper()
    big_model = BigModel()
    text_audio_converter = TextToSpeech()
    listener = KeyboardListener(recorder, audio_text_converter, big_model, text_audio_converter)
    listener.start()
    print("Hold 'Enter' to record, press 'q' to quit.")
    listener.join()


if __name__ == '__main__':
    main()


# class Recorder:
#     def __init__(self):
#         self.frames = None
#         self.streamframes = None
#         self.chunk = 1024  # Record in chunks of 1024 samples
#         self.sample_format = pyaudio.paInt16  # 16 bits per sample
#         self.channels = 2
#         self.fs = 44100  # Record at 44100 samples per second
#         self.seconds = 3
#         self.filename = "output.wav"
#         self.p = pyaudio.PyAudio()  # Create an interface to PortAudio
#         self.stream = None

    # def record(self):
    #     print('Recording')
    #     self.stream = self.p.open(format=self.sample_format,
    #                               channels=self.channels,
    #                               rate=self.fs,
    #                               frames_per_buffer=self.chunk,
    #                               input=True)
    #
    #     self.frames = []  # Initialize array to store frames
    #
    #     # Store data in chunks for 3 seconds
    #     for i in range(0, int(self.fs / self.chunk * self.seconds)):
    #         data = self.stream.read(self.chunk)
    #         self.frames.append(data)
    #
    #     # Stop and close the stream
    #     self.stream.stop_stream()
    #     self.stream.close()
    #     # Terminate the PortAudio interface
    #     # self.p.terminate()
    #
    #     print('Finished recording')
    #
    #     # Save the recorded data as a WAV file
    #     wf = wave.open(self.filename, 'wb')
    #     wf.setnchannels(self.channels)
    #     wf.setsampwidth(self.p.get_sample_size(self.sample_format))
    #     wf.setframerate(self.fs)
    #     wf.writeframes(b''.join(self.frames))
    #     wf.close()