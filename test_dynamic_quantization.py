import torch
import librosa
import soundfile as sf
from lightning_module import CodecLightningModule

def test_dynamic_quantization():
    model = CodecLightningModule.load_from_checkpoint("/mnt/f/PyCharmProjects/X-Codec2/models/epoch=4-step=1400000.ckpt")
    model.eval()

    wav_simple = librosa.load("simple_audio.wav", sr=16000)[0]
    wav_complex = librosa.load("complex_audio.wav", sr=16000)[0]

    wav_simple = torch.from_numpy(wav_simple).unsqueeze(0).cuda()
    wav_complex = torch.from_numpy(wav_complex).unsqueeze(0).cuda()

    with torch.no_grad():
        vq_simple = model.CodecEnc(wav_simple.unsqueeze(1))
        complexity_simple = model.compute_audio_complexity(vq_simple)
        print(f"Complexity score for simple audio: {complexity_simple.item()}")

        vq_complex = model.CodecEnc(wav_complex.unsqueeze(1))
        complexity_complex = model.compute_audio_complexity(vq_complex)
        print(f"Complexity score for complex audio: {complexity_complex.item()}")

def test_quantizer_selection():
    model = CodecLightningModule.load_from_checkpoint("/mnt/f/PyCharmProjects/X-Codec2/models/epoch=4-step=1400000.ckpt")
    model.eval()

    test_features = torch.randn(1, 2048, 100).cuda()
    complexity_scores = [0.2, 0.5, 0.8]

    for score in complexity_scores:
        vq_post_emb, vq_code, vq_loss = model.generator(test_features, vq=True, complexity_score=score)
        print(f"Complexity score: {score}, VQ Code shape: {vq_code.shape}, Loss: {vq_loss.item()}")

def compare_quantization_methods():  
    model = CodecLightningModule.load_from_checkpoint("/mnt/f/PyCharmProjects/X-Codec2/models/epoch=4-step=1400000.ckpt")  
    model.eval()  
      
    test_audios = ["test1.wav", "test2.wav", "test3.wav"]  
      
    for audio_path in test_audios:  
        wav = librosa.load(audio_path, sr=16000)[0]  
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).cuda()  
          
        with torch.no_grad():  
            vq_emb = model.CodecEnc(wav_tensor.unsqueeze(1))  
            vq_post_emb_fixed, _, _ = model.generator(vq_emb, vq=True, complexity_score=None)  
            fixed_recon = model.generator(vq_post_emb_fixed, vq=False)  
              
            complexity_score = model.compute_audio_complexity(vq_emb.transpose(1, 2))  
            vq_post_emb_dynamic, _, _ = model.generator(vq_emb, vq=True, complexity_score=complexity_score)  
            dynamic_recon = model.generator(vq_post_emb_dynamic, vq=False)  
          
        sf.write(f"fixed_{audio_path}", fixed_recon.squeeze().cpu().numpy(), 16000)  
        sf.write(f"dynamic_{audio_path}", dynamic_recon.squeeze().cpu().numpy(), 16000)  
        print(f"Audio {audio_path}: complexity_score = {complexity_score.item():.4f}")

if __name__ == "__main__":
    test_dynamic_quantization()
    test_quantizer_selection()
    compare_quantization_methods()
    print("Dynamic quantization tests completed successfully.")