FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y libportaudio2 software-properties-common git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/w-okada/voice-changer.git /app
RUN cd app
WORKDIR /app
RUN pip install --no-cache-dir faiss-gpu fairseq pyngrok
RUN pip install --no-cache-dir pyworld --no-build-isolation
RUN pip install --no-cache-dir -r requirements.txt
RUN !pip uninstall -y torch torchaudio
RUN !pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
EXPOSE 8000
CMD ["python3", "MMVCServerSIO.py", \
     "-p", "8000", \
     "--https", "False", \
     "--content_vec_500", "pretrain/checkpoint_best_legacy_500.pt", \
     "--content_vec_500_onnx", "pretrain/content_vec_500.onnx", \
     "--content_vec_500_onnx_on", "true", \
     "--hubert_base", "pretrain/hubert_base.pt", \
     "--hubert_base_jp", "pretrain/rinna_hubert_base_jp.pt", \
     "--hubert_soft", "pretrain/hubert/hubert-soft-0d54a1f4.pt", \
     "--nsf_hifigan", "pretrain/nsf_hifigan/model", \
     "--crepe_onnx_full", "pretrain/crepe_onnx_full.onnx", \
     "--crepe_onnx_tiny", "pretrain/crepe_onnx_tiny.onnx", \
     "--rmvpe", "pretrain/rmvpe.pt", \
     "--model_dir", "model_dir", \
     "--samples", "samples.json"]