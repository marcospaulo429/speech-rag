"""SLM Generator: Gerador baseado em Speech Language Model."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from typing import Union, List, Optional, Dict, Any
import yaml

from .audio_conditioning import AudioConditioner


class SLMGenerator:
    """Gerador baseado em Speech Language Model condicionado em áudio."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Audio-Chat",
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        device: Optional[str] = None,
        freeze: bool = True
    ):
        """
        Inicializa o gerador SLM.
        
        Args:
            model_name: Nome do modelo no Hugging Face
            max_length: Comprimento máximo de geração
            temperature: Temperatura para sampling
            do_sample: Se True, usa sampling; se False, usa greedy
            device: Device para processamento
            freeze: Se True, congela parâmetros (geralmente True para inferência)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega modelo e processador
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except:
            # Fallback para modelos que não são CausalLM
            self.model = AutoModel.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)
        
        # Congela se solicitado
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Carrega processador/tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.processor = None
            except:
                self.processor = None
                self.tokenizer = None
        
        # Audio conditioner
        self.audio_conditioner = AudioConditioner(
            max_audio_length=30.0,  # 30 segundos padrão
            sample_rate=16000
        )
    
    @classmethod
    def from_config(cls, config_path: str, device: Optional[str] = None):
        """
        Cria gerador a partir de configuração.
        
        Args:
            config_path: Caminho para arquivo YAML
            device: Device
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            model_name=config.get("model_name", "Qwen/Qwen-Audio-Chat"),
            max_length=config.get("max_length", 512),
            temperature=config.get("temperature", 0.7),
            do_sample=config.get("do_sample", True),
            device=device,
            freeze=True
        )
    
    def generate(
        self,
        audio: Union[torch.Tensor, List[torch.Tensor], str],
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Gera resposta condicionada em áudio.
        
        Args:
            audio: Áudio para condicionamento
            prompt: Prompt de texto opcional
            max_length: Comprimento máximo (None = usa self.max_length)
            temperature: Temperatura (None = usa self.temperature)
        
        Returns:
            Texto gerado
        """
        self.model.eval()
        
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        
        # Prepara áudio
        prepared_audio = self.audio_conditioner.prepare_audio(audio)
        
        # Formata entrada
        inputs = self.audio_conditioner.format_for_slm(prepared_audio, prompt)
        
        # Processa com processor se disponível
        if self.processor is not None:
            try:
                # Tenta processar áudio e texto juntos
                if isinstance(prepared_audio, list):
                    # Múltiplos chunks - pega o primeiro ou concatena
                    prepared_audio = prepared_audio[0] if len(prepared_audio) > 0 else prepared_audio
                
                processed = self.processor(
                    text=prompt,
                    audio=prepared_audio,
                    return_tensors="pt",
                    padding=True
                )
                processed = {k: v.to(self.device) for k, v in processed.items()}
            except Exception as e:
                # Fallback: processa apenas texto
                if prompt is not None and self.tokenizer is not None:
                    processed = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True
                    )
                    processed = {k: v.to(self.device) for k, v in processed.items()}
                else:
                    raise ValueError(f"Erro ao processar entrada: {e}")
        elif self.tokenizer is not None and prompt is not None:
            # Apenas texto
            processed = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            )
            processed = {k: v.to(self.device) for k, v in processed.items()}
        else:
            raise ValueError("Processor ou tokenizer não disponível")
        
        # Gera
        with torch.no_grad():
            outputs = self.model.generate(
                **processed,
                max_length=max_length,
                temperature=temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None
            )
        
        # Decodifica
        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif self.processor is not None:
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_text = str(outputs[0])
        
        return generated_text
    
    def generate_batch(
        self,
        audio_list: List[Union[torch.Tensor, str]],
        prompts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Gera respostas para um batch de áudios.
        
        Args:
            audio_list: Lista de áudios
            prompts: Lista de prompts (None = sem prompt)
        
        Returns:
            Lista de textos gerados
        """
        if prompts is None:
            prompts = [None] * len(audio_list)
        
        results = []
        for audio, prompt in zip(audio_list, prompts):
            result = self.generate(audio, prompt)
            results.append(result)
        
        return results

