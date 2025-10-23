#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# voicevox_ros2.py

import rclpy
from rclpy.node import Node
import subprocess
import threading
from std_msgs.msg import String, UInt8MultiArray
from voicevox_ros2_msgs.srv import Speaking
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
from voicevox_core import StyleNotFoundError

class VoiceVoxRos2(Node):
    def __init__(self):
        super().__init__('voicevox_ros2')

        self.declare_parameter('voicevox_onnxruntime_path', '/ws/voicevox_core/onnxruntime/lib/')
        self.declare_parameter('voicevox_model_path', '/ws/voicevox_core/models/vvms/0.vvm')
        self.declare_parameter('open_jtalk_dict_dir', '/ws/voicevox_core/dict/open_jtalk_dic_utf_8-1.11')

        voicevox_onnxruntime_path = self.get_parameter('voicevox_onnxruntime_path').value + Onnxruntime.LIB_VERSIONED_FILENAME
        voicevox_model_path = self.get_parameter('voicevox_model_path').value
        open_jtalk_dict_dir = self.get_parameter('open_jtalk_dict_dir').value

        self.get_logger().info('voicevox_core parameters: \n    %s\n    %s\n    %s'%(voicevox_onnxruntime_path, voicevox_model_path, open_jtalk_dict_dir))

        self.synthesizer = Synthesizer(Onnxruntime.load_once(filename=voicevox_onnxruntime_path), OpenJtalk(open_jtalk_dict_dir))
        with VoiceModelFile.open(voicevox_model_path) as model:
            self.get_logger().info('load model: %s'%voicevox_model_path)
            self.synthesizer.load_voice_model(model)
        
        # --- /speak サービス ---
        self.speaking_service = self.create_service(
            Speaking,
            'speak',
            self.speaking_cb
        )

        # 1. Web UI 向けのオーディオ配信用パブリッシャー
        self.audio_pub = self.create_publisher(UInt8MultiArray, '/tts_audio_data', 10)
        
        # 2. チャットボットのストリームテキスト用サブスクライバー
        self.stream_sub = self.create_subscription(
            String,
            '/chatbot_response_stream',
            self.stream_callback,
            10)
        
        # 3. テキストバッファリングの仕組み
        self._lock = threading.Lock() # 処理の重複実行を防ぐロック
        self._sentence_buffer = ""
        self._buffer_lock = threading.Lock()
        self._speech_timer = None
        self._SPEECH_TIMEOUT = 0.5 # 0.5秒間新しいテキストが来なければ発話

        self.get_logger().info('🔊 VoiceVox ROS2 Node (WebUI Stream Ready) Start!')
    

    def _generate_and_publish_audio(self, text: str, speaker_id: int = 3, pitch_scale: float = 0.0, intonation_scale: float = 0.0, speed_scale: float = 0.0, volume_scale: float = 0.0, enable_interrogative_upspeak: bool = True) -> bool:
        """
        [新設] テキストから音声を生成し、/tts_audio_data トピックに発行する共通関数
        """
        if self._lock.locked():
            self.get_logger().warn('Audio generation is already in progress. Skipping.')
            return False
        
        with self._lock:
            try:
                self.get_logger().info(f'Generating TTS for: "{text}"')
                audio_query = self.synthesizer.create_audio_query(text, speaker_id)
                audio_query.pitch_scale += pitch_scale
                audio_query.intonation_scale += intonation_scale
                audio_query.speed_scale += speed_scale
                audio_query.volume_scale += volume_scale
                wav = self.synthesizer.synthesis(audio_query, speaker_id, enable_interrogative_upspeak=enable_interrogative_upspeak)
                
                if wav:
                    self.get_logger().info('Successfully generate speech !')
                    
                    # Web UI 向けにトピック発行
                    ros_msg = UInt8MultiArray()
                    ros_msg.data = list(wav)
                    self.audio_pub.publish(ros_msg)
                    self.get_logger().info(f'Published {len(wav)} bytes of audio data to /tts_audio_data.')
                    
                    return True
            
            except StyleNotFoundError as e:
                self.get_logger().error(str(e))
            
            except Exception as e:
                self.get_logger().error(f'An error occurred during TTS synthesis: {e}')

        return False

    def speaking_cb(self, req:Speaking.Request, res:Speaking.Response):
        """
        [修正] /speak サービスが呼ばれた時も、ローカル再生(aplay)せず、
        Web UI (/tts_audio_data) に音声を発行するように変更
        """
        self.get_logger().info('Request (from /speak service):\n  text: %s\n  speaker_id: %d...'%(req.text, req.speaker_id))
        
        res.success = self._generate_and_publish_audio(
            text=req.text,
            speaker_id=req.speaker_id,
            pitch_scale=req.pitch_scale,
            intonation_scale=req.intonation_scale,
            speed_scale=req.speed_scale,
            volume_scale=req.volume_scale,
            enable_interrogative_upspeak=req.enable_interrogative_upspeak
        )
        
        return res

    def stream_callback(self, msg: String):
        """
        [新設] /chatbot_response_stream を受信するたびに呼び出される
        """
        with self._buffer_lock:
            self._sentence_buffer += msg.data

            # 既存のタイマーがあればキャンセル
            if self._speech_timer is not None:
                self._speech_timer.cancel()
            
            # 新しいタイマーを設定
            self._speech_timer = threading.Timer(self._SPEECH_TIMEOUT, self._process_and_speak_buffer)
            self._speech_timer.start()

    def _process_and_speak_buffer(self):
        """
        バッファ内のテキストを音声化する
        """
        text_to_speak = ""
        with self._buffer_lock:
            text_to_speak = self._sentence_buffer.strip()
            self._sentence_buffer = ""

        if not text_to_speak:
            return

        # [変更] (スピーカーIDは3=ずんだもんに固定)
        self._generate_and_publish_audio(text_to_speak, speaker_id=3)


def main():
    rclpy.init()
    node = VoiceVoxRos2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()