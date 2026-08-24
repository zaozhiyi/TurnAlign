# Acknowledgements / 致谢

TurnAlign is assembled from established ideas and open-source tooling in speech
recognition. We list both runtime integrations and design references here, even
when no source code was copied. Upstream projects, model weights, and datasets
keep their own licenses; users must review the applicable upstream terms.

TurnAlign 建立在语音识别社区已有的工程和方法之上。这里同时列出运行时集成与设计参考，
即使没有复制对方源代码也会注明。上游代码、模型权重和数据集继续遵循各自许可证，使用者
需要自行确认对应条款。

## Runtime integrations / 运行时集成

| Project | Role in TurnAlign | Upstream license |
|---|---|---|
| [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | Optional Chinese/English/Cantonese ASR backend | MIT model card |
| [OpenAI Whisper](https://github.com/openai/whisper) | Whisper model family used by several optional backends | MIT |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Optional GLM-ASR and Whisper inference adapter | Apache-2.0 |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Optional CTranslate2 Whisper backend | MIT |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Optional command-line Whisper backend | MIT |
| [FunASR](https://github.com/modelscope/FunASR) | Validation pipeline for FSMN-VAD, Paraformer timing and CAM++ speaker embeddings | MIT toolkit; model terms vary |
| [PyTorch](https://github.com/pytorch/pytorch) | CUDA, ROCm and MPS inference runtime used by model plugins | BSD-style |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Portable inference provider target | MIT |
| [FFmpeg](https://ffmpeg.org/) | Local decoding of MP3, M4A and other audio formats | LGPL/GPL depending on build |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) | Optional cross-platform microphone capture | MIT |
| [websockets](https://github.com/python-websockets/websockets) | Optional WebSocket transport | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | PCM conversion in optional Python model backends | BSD-3-Clause |
| [UMAP](https://github.com/lmcinnes/umap) | Long-recording speaker embedding projection in validation experiments | BSD-3-Clause |
| [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) | Long-recording speaker clustering in validation experiments | BSD-3-Clause |

## Design and implementation references / 设计与实现参考

| Project | What we learned from it |
|---|---|
| [Whisper-Streaming](https://github.com/ufal/whisper_streaming) | LocalAgreement, revisable tails, adaptive latency, VAD/VAC and long-audio simulation |
| [SimulStreaming](https://github.com/ufal/SimulStreaming) | Model warm-up, microphone-over-TCP, file simulation and incremental policies |
| [WhisperLive](https://github.com/collabora/WhisperLive) | Raw PCM WebSocket sessions, microphone/file parity, partial and committed callbacks, optional diarization |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Portable streaming/non-streaming interfaces, microphone examples and broad platform support |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Streaming voice activity control and endpointing patterns |
| [WhisperX](https://github.com/m-bain/whisperX) | Separating ASR text, forced alignment and speaker diarization tracks |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker embeddings, diarization and offline refinement patterns |

TurnAlign's current dependency-free energy endpoint detector is original project
code, but its placement in the pipeline follows common VAD/endpointing practice
demonstrated by the projects above. The common event names and wire format are
TurnAlign-specific and are not claimed as a new ASR algorithm.

If an attribution is missing or inaccurate, please open an issue or pull request.
如有遗漏或描述不准确，欢迎提交 issue 或 pull request。
