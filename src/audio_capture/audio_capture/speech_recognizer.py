import json
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
from vosk import Model, KaldiRecognizer


class SpeechRecognizer(Node):
    def __init__(self):
        super().__init__('speech_recognizer')

        self.declare_parameter('model_path', '/home/kneepolean/AUTOKnee-server/models/vosk-model-small-en-us-0.15')
        self.declare_parameter('audio_topic', '/audio')
        self.declare_parameter('text_topic', '/speech')
        self.declare_parameter('input_sample_rate', 44100)
        self.declare_parameter('input_channels', 2)

        model_path = self.get_parameter('model_path').value
        audio_topic = self.get_parameter('audio_topic').value
        text_topic = self.get_parameter('text_topic').value
        self.input_rate = self.get_parameter('input_sample_rate').value
        self.input_channels = self.get_parameter('input_channels').value

        # Vosk works best at 16kHz mono
        self.target_rate = 16000

        self.get_logger().info(f"Loading Vosk model from {model_path}...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.target_rate)
        self.get_logger().info("Vosk model loaded.")

        self.publisher_ = self.create_publisher(String, text_topic, 10)
        self.subscription = self.create_subscription(
            Int16MultiArray, audio_topic, self._audio_callback, 10
        )

        self.get_logger().info(
            f"Listening on '{audio_topic}', publishing recognized text on '{text_topic}'"
        )

    def _downsample(self, audio_int16):
        """Convert stereo 44100Hz int16 to mono 16kHz int16."""
        audio = audio_int16.astype(np.float32)

        # Stereo to mono
        if self.input_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample from input_rate to 16kHz via linear interpolation
        if self.input_rate != self.target_rate:
            ratio = self.target_rate / self.input_rate
            n_samples = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, n_samples)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.int16)

    def _audio_callback(self, msg):
        audio_int16 = np.array(msg.data, dtype=np.int16)
        downsampled = self._downsample(audio_int16)
        audio_bytes = downsampled.tobytes()

        if self.recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self.recognizer.Result())
            text = result.get('text', '').strip()
            if text:
                self.get_logger().info(f"Recognized: {text}")
                out = String()
                out.data = text
                self.publisher_.publish(out)
        else:
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get('partial', '').strip()
            if partial_text:
                self.get_logger().debug(f"Partial: {partial_text}")


def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
