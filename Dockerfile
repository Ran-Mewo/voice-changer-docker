FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

#ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y g++ gcc libportaudio2 git
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/w-okada/voice-changer.git /app
WORKDIR /app/server
RUN pip install --no-cache-dir faiss-gpu fairseq pyngrok
RUN pip install --no-cache-dir pyworld --no-build-isolation
RUN grep -v 'torch\|torch-audio' requirements.txt > requirements_no_torch.txt
RUN pip install --no-cache-dir -r requirements_no_torch.txt
RUN rm requirements_no_torch.txt
RUN pip install torchcrepe torchfcpe --no-cache-dir --no-deps
RUN pip uninstall onnxruntime-gpu -y
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ --no-cache-dir
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