import speech_recognition as sr
import google.generativeai as genai


# Replace with your API Key
API_KEY = "AIzaSyCKZwZL4gx3Kj-Yeho8YXi5DcfjBOp2UWs"

genai.configure(
  api_key=API_KEY
)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def listen_user():
  """
  Listens to user input using microphone and returns recognized text.
  """
  recognizer = sr.Recognizer()
  with sr.Microphone() as source:
    print("Listening...")
    audio = recognizer.listen(source)
  try:
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
    return text
  except sr.UnknownValueError:
    print("Sorry, could not understand audio")
    return None
  except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return None

def speak_response(text):
  """
  Uses a text-to-speech engine to speak the response to the user (implementation not included).
  You'll need an external library for this functionality. 
  """
  # Replace this with your text-to-speech library implementation
  print("Gemini says:", text)

def chat_with_gemini():
  """
  Main chat loop that takes user input, sends it to Gemini for generation, and speaks the response.
  """
  client = ApiClient(configuration=auth.ApiKeyAuth(api_key={"api_key": API_KEY}))
  while True:
    user_input = listen_user()
    if user_input is None:
      continue
    # Craft the request with the user input as prompt
    request = GenerateTextRequest(prompt=user_input, max_tokens=100)
    # Send the request and get the generated response
    response = client.projects().endpoints().generateText(request=request)
    generated_text = response.text.split("\n\n")[0]  # Assuming first part is response
    speak_response(generated_text)

if __name__ == "__main__":
  print("Starting voice chat with Gemini...")
  chat_with_gemini()
