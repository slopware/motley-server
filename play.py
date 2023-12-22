import pyaudio
import wave

def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create an instance of PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # Read data in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    # Play the audio
    while data != b'':
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Replace 'path_to_your_audio_file.wav' with the path to your audio file
play_audio('test_output.mp3')
