import base64
from threading import Lock, Thread
import cv2
import numpy
import openai
import pyautogui
import os
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"API key loaded: {'Yes' if api_key else 'No'}")

class DesktopStream:
    def __init__(self):
        self.running = False
        self.lock = Lock()
        # Capture initial screenshot
        self.frame = pyautogui.screenshot()
        self.frame = cv2.cvtColor(numpy.array(self.frame), cv2.COLOR_RGB2BGR)

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screen = pyautogui.screenshot()
            frame = cv2.cvtColor(numpy.array(screen), cv2.COLOR_RGB2BGR)
            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return
        print("Prompt:", prompt)
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()
        print("Response:", response)
        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions about what's happening on the user's desktop.
        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 
        Be friendly and helpful. Show some personality.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )
        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

desktop_stream = DesktopStream().start()
model = ChatOpenAI(model="gpt-4")
assistant = Assistant(model)

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, desktop_stream.read(encode=True))
    except UnknownValueError:
        print("There was an error processing the audio.")

recognizer = Recognizer()
microphone = Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("desktop", desktop_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

desktop_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)