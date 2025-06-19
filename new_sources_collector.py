import json
import logging
import hashlib
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

@dataclass
class ModelSignature:
    """EXACT same structure as your original collector"""
    id: str
    prompt: str
    response: str
    cleaned_response: str
    model_name: str
    provider: str
    quality_score: float
    artifact_removal_score: float
    source_dataset: str
    preprocessing_steps: List[str]
    response_length: int
    word_count: int
    lexical_diversity: float
    passes_filters: bool
    timestamp: str
    content_hash: str

class NewSourcesCollector:
    """Focused collector for new high-value sources - TOP 3 NON-OPENAI DATASETS"""
    
    def __init__(self, target_per_model: int = 2000, min_quality_score: float = 0.5):
        self.target_per_model = target_per_model
        self.min_quality_score = min_quality_score
        # No hash tracking - combiner handles all deduplication
        
        # HIGH-VALUE SOURCES + GATED DATASETS (with HF token)
        self.new_sources = {
            # WORKING PUBLIC DATASETS
            'hc3': {
                'dataset': 'Hello-SimpleAI/HC3',
                'split': 'train',
                'config': 'all',
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'format': 'hc3',
                'max_samples': 15000,
                'gated': False
            },
            'wildchat_1m': {
                'dataset': 'allenai/WildChat-1M',
                'split': 'train',
                'model': 'various',
                'provider': 'openai',
                'format': 'wildchat',
                'max_samples': 20000,
                'gated': False
            },
            'sharegpt_gpt4': {
                'dataset': 'shibing624/sharegpt_gpt4',
                'split': 'train',
                'model': 'gpt-4',
                'provider': 'openai',
                'format': 'sharegpt',
                'max_samples': 6000,
                'gated': False
            },
            
            # TOP NON-OPENAI DATASETS
            'lmsys_chat_1m': {
                'dataset': 'lmsys/lmsys-chat-1m',
                'split': 'train',
                'model': 'various',  # Multi-provider: Meta, Google, Anthropic, etc.
                'provider': 'various',
                'format': 'lmsys_chat',
                'max_samples': 25000,
                'gated': True
            },
            'chatbot_arena': {
                'dataset': 'lmsys/chatbot_arena_conversations',
                'split': 'train', 
                'model': 'various',  # Multiple models including Llama, Claude, Gemini
                'provider': 'various',
                'format': 'arena',
                'max_samples': 15000,
                'gated': False
            },
            'open_orca': {
                'dataset': 'Open-Orca/OpenOrca',
                'split': 'train',
                'model': 'various',  # Mistral, Llama, and other open models
                'provider': 'various', 
                'format': 'orca',
                'max_samples': 20000,
                'gated': False
            },
            
            # GATED HIGH-VALUE DATASETS (NOW ACCESSIBLE)
            'anthropic_hh_rlhf': {
                'dataset': 'Anthropic/hh-rlhf',
                'split': 'train',
                'model': 'claude',
                'provider': 'anthropic',
                'format': 'anthropic_hh',
                'max_samples': 20000,
                'gated': True
            },
            'oasst1': {
                'dataset': 'OpenAssistant/oasst1',
                'split': 'train',
                'model': 'various',  # Multiple open models
                'provider': 'various',
                'format': 'oasst',
                'max_samples': 15000,
                'gated': False
            }
        }
        
        # Enhanced model patterns for new datasets
        self.model_patterns = [
            (re.compile(r'gpt-?4o', re.I), 'gpt-4o'),
            (re.compile(r'gpt-?4', re.I), 'gpt-4'),
            (re.compile(r'gpt-?3\.5|chatgpt', re.I), 'gpt-3.5-turbo'),
            (re.compile(r'claude-?3\.5', re.I), 'claude-3.5'),
            (re.compile(r'claude-?3', re.I), 'claude-3'),
            (re.compile(r'claude-?2', re.I), 'claude-2'),
            (re.compile(r'claude(?!-[23])', re.I), 'claude'),
            (re.compile(r'gemini.*pro', re.I), 'gemini-pro'),
            (re.compile(r'gemini|bard', re.I), 'gemini'),
            (re.compile(r'palm-?2', re.I), 'palm-2'),
            (re.compile(r'perplexity', re.I), 'perplexity'),
            (re.compile(r'llama-?3\.1', re.I), 'llama-3.1'),
            (re.compile(r'llama-?3', re.I), 'llama-3'),
            (re.compile(r'llama-?2', re.I), 'llama-2'),
            (re.compile(r'llama(?!-[23])', re.I), 'llama'),
            (re.compile(r'vicuna', re.I), 'vicuna'),
            (re.compile(r'mistral.*7b', re.I), 'mistral-7b'),
            (re.compile(r'mistral', re.I), 'mistral'),
        ]

    def collect_new_signatures(self, output_path: str = "new_sources_signatures.json") -> Dict[str, Any]:
        """Main collection method - NO deduplication (handled by combiner)"""
        logger.info(f"🚀 Processing {len(self.new_sources)} high-value sources (including GATED datasets)")
        logger.info("📊 Non-OpenAI: LMSYS-Chat-1M, Chatbot Arena, OpenOrca, Anthropic HH-RLHF, OpenAssistant")
        logger.info("🔐 GATED: LMSYS-Chat-1M, Anthropic HH-RLHF")
        all_signatures = []
        collection_stats = {}
        
        for source_name, config in self.new_sources.items():
            logger.info(f"Processing: {source_name}")
            try:
                signatures = self._process_new_source(source_name, config)
                if signatures:
                    all_signatures.extend(signatures)
                    model_counts = Counter(sig.model_name for sig in signatures)
                    for model, count in model_counts.items():
                        collection_stats[model] = collection_stats.get(model, 0) + count
                    logger.info(f"✅ {source_name}: {len(signatures)} signatures")
            except Exception as e:
                logger.error(f"❌ {source_name}: {e}")
        
        if not all_signatures:
            return {'signatures': [], 'report': {}, 'total_samples': 0}
        
        # Only basic filtering - deduplication happens in combiner
        cleaned = self._clean_and_filter(all_signatures)
        logger.info(f"After basic filtering: {len(cleaned)}")
        
        # Generate report (no deduplication stats)
        report = self._generate_report(cleaned, collection_stats)
        self._save_dataset(cleaned, report, output_path)
        
        return {
            'signatures': cleaned,
            'report': report,
            'total_samples': len(cleaned)
        }

    def _process_new_source(self, source_name: str, config: Dict) -> List[ModelSignature]:
        """Process individual new source with HuggingFace token support"""
        signatures = []
        
        try:
            # Handle HuggingFace datasets with configs and authentication
            load_kwargs = {
                'split': config.get('split', 'train'),
                'streaming': True,
                'trust_remote_code': True
            }
            
            # Add HuggingFace token for gated datasets
            if config.get('gated', False):
                load_kwargs['token'] = True  # Uses HF_TOKEN environment variable or cached token
                logger.info(f"Using HuggingFace token for gated dataset: {source_name}")
            
            # Add config name if specified (like HC3 has multiple configs)
            if 'config' in config:
                dataset = load_dataset(
                    config['dataset'],
                    config['config'],  # e.g., 'all' for HC3
                    **load_kwargs
                )
            else:
                dataset = load_dataset(
                    config['dataset'],
                    **load_kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to load {source_name}: {e}")
            if config.get('gated', False):
                logger.error(f"Hint: Make sure you're authenticated with HuggingFace CLI: huggingface-cli login")
            return []
        
        processed = 0
        max_items = config.get('max_samples', 15000)
        
        for item in dataset:
            if len(signatures) >= self.target_per_model or processed >= max_items:
                break
                
            try:
                prompt, response = self._extract_content_new(item, config)
                if self._basic_quality_check(prompt, response):
                    model_name = self._extract_model_name(item, config)
                    provider = self._get_provider_from_model(model_name)
                    
                    signature = self._create_signature(
                        prompt=prompt,
                        response=response,  
                        model_name=model_name,
                        provider=provider,
                        source_dataset=source_name,
                        index=processed
                    )
                    if signature:
                        signatures.append(signature)
            except Exception:
                continue
            processed += 1
            
            if processed % 2000 == 0:
                logger.info(f"   Processed {processed} items, collected {len(signatures)} signatures")
        
        return signatures

    def _extract_content_new(self, item: Dict, config: Dict) -> Tuple[str, str]:
        """Extract content from new sources - ENHANCED with ALL formats including gated datasets"""
        format_type = config['format']
        
        if format_type == 'lmsys_chat':
            # LMSYS-Chat-1M format
            conv = item.get('conversation', [])
            if len(conv) >= 2:
                for i in range(len(conv) - 1):
                    if (conv[i].get('role') == 'user' and 
                        conv[i + 1].get('role') == 'assistant'):
                        return conv[i].get('content', ''), conv[i + 1].get('content', '')
            return '', ''
            
        elif format_type == 'arena':
            # Chatbot Arena format
            conv_a = item.get('conversation_a', [])
            if len(conv_a) >= 2:
                for i in range(len(conv_a) - 1):
                    if (conv_a[i].get('role') == 'user' and 
                        conv_a[i + 1].get('role') == 'assistant'):
                        return conv_a[i].get('content', ''), conv_a[i + 1].get('content', '')
            return '', ''
            
        elif format_type == 'orca':
            # OpenOrca format
            system_prompt = item.get('system_prompt', '')
            question = item.get('question', '')
            response = item.get('response', '')
            
            # Combine system prompt and question for full context
            if system_prompt and question:
                full_prompt = f"{system_prompt}\n\n{question}"
            elif question:
                full_prompt = question
            else:
                full_prompt = system_prompt
                
            return full_prompt, response
            
        elif format_type == 'anthropic_hh':
            # Anthropic HH-RLHF format
            chosen = item.get('chosen', '') or item.get('response', '')
            prompt = item.get('prompt', '') or item.get('question', '')
            
            if not prompt and chosen:
                # Try to split if it's in Human/Assistant format
                parts = chosen.split('\n\nAssistant:')
                if len(parts) >= 2:
                    prompt = parts[0].replace('Human:', '').strip()
                    response = parts[1].split('\n\nHuman:')[0].strip()
                    return prompt, response
            
            return prompt, chosen
            
        elif format_type == 'oasst':
            # OpenAssistant format
            text = item.get('text', '')
            parent_id = item.get('parent_id', '')
            message_id = item.get('message_id', '')
            role = item.get('role', '')
            
            # For now, treat each message as standalone
            # In a more sophisticated approach, you'd reconstruct conversation trees
            if role == 'assistant' and text:
                # Use a generic prompt for assistant responses
                return "Provide a helpful response.", text
            elif role == 'prompter' and text:
                # This is a user prompt, would need to find corresponding assistant response
                return text, "I'll help you with that."
            
            return '', ''
            
        elif format_type == 'sharegpt':
            # ShareGPT format (works for both regular and gated ShareGPT datasets)
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                for i in range(len(conversations) - 1):
                    if (conversations[i].get('from') == 'human' and
                        conversations[i + 1].get('from') == 'gpt'):
                        return conversations[i].get('value', ''), conversations[i + 1].get('value', '')
            return '', ''
            
        elif format_type == 'wildchat':
            # WildChat format
            conv = item.get('conversation', [])
            if len(conv) >= 2:
                for i in range(len(conv) - 1):
                    if (conv[i].get('role') == 'user' and 
                        conv[i + 1].get('role') == 'assistant'):
                        return conv[i].get('content', ''), conv[i + 1].get('content', '')
            return '', ''
            
        elif format_type == 'hc3':
            # HC3 dataset structure: question, chatgpt_answers, human_answers
            question = item.get('question', '')
            chatgpt_answers = item.get('chatgpt_answers', [])
            
            if question and chatgpt_answers:
                # Take the first ChatGPT answer (they usually have multiple)
                chatgpt_response = chatgpt_answers[0] if isinstance(chatgpt_answers, list) and len(chatgpt_answers) > 0 else str(chatgpt_answers)
                if chatgpt_response and len(chatgpt_response) > 50:
                    return question, chatgpt_response
            
            return '', ''
            
        return '', ''

    def _extract_model_name(self, item: Dict, config: Dict) -> str:
        """Extract model name - handles various formats including gated datasets"""
        if config['model'] != 'various':
            return config['model']
            
        # Try to extract from item based on format
        format_type = config['format']
        
        if format_type == 'lmsys_chat':
            # LMSYS datasets have model field
            model = item.get('model', '') or item.get('model_name', '')
        elif format_type == 'arena':
            # Arena has model_a, model_b
            model = item.get('model_a', '') or item.get('model_b', '') or item.get('model', '')
        elif format_type == 'orca':
            # OpenOrca might not have explicit model field - infer from dataset
            model = 'mistral-7b'  # Most OpenOrca models are Mistral-based
        elif format_type == 'anthropic_hh':
            # Anthropic HH-RLHF is Claude models
            model = 'claude'
        elif format_type == 'oasst':
            # OpenAssistant uses various models, try to extract or default
            model = item.get('model_name', '') or item.get('model', '') or 'assistant'
        else:
            # Generic extraction
            model = (item.get('model', '') or 
                    item.get('model_name', '') or
                    item.get('generator', '') or
                    'unknown')
        
        return self._normalize_model_name(model)

    def _normalize_model_name(self, model_name: str) -> str:
        """Enhanced normalization for new model types"""
        if not model_name or model_name == 'unknown':
            return 'unknown'
        model_name = model_name.strip()
        for pattern, normalized_name in self.model_patterns:
            if pattern.search(model_name):
                return normalized_name
        return model_name.lower()

    def _get_provider_from_model(self, model_name: str) -> str:
        """Enhanced provider mapping for new datasets"""
        model_name = model_name.lower()
        if any(x in model_name for x in ['gpt', 'davinci', 'chatgpt']):
            return 'openai'
        elif 'claude' in model_name:
            return 'anthropic'
        elif any(x in model_name for x in ['gemini', 'bard', 'palm']):
            return 'google'
        elif 'perplexity' in model_name:
            return 'perplexity'
        elif any(x in model_name for x in ['llama', 'vicuna']):
            return 'meta'
        elif 'mistral' in model_name:
            return 'mistral'
        return 'unknown'

    def _basic_quality_check(self, prompt: str, response: str) -> bool:
        """Same quality check as your original"""
        if not prompt or not response:
            return False
        if len(response) < 50 or len(response) > 5000:
            return False
        if len(response.split()) < 5:
            return False
        return True

    def _create_signature(self, prompt: str, response: str, model_name: str,
                         provider: str, source_dataset: str, index: int) -> Optional[ModelSignature]:
        """Create signature - NO hash tracking (combiner handles deduplication)"""
        try:
            cleaned_response = self._clean_text(response)
            content_hash = hashlib.md5(cleaned_response.encode()).hexdigest()
            
            words = cleaned_response.split()
            word_count = len(words)
            lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
            quality_score = self._calculate_quality_score(cleaned_response)
            artifact_score = self._calculate_artifact_removal_score(response, cleaned_response)
            
            signature = ModelSignature(
                id=f"{source_dataset}_{index}_{content_hash[:8]}",
                prompt=prompt,
                response=response,
                cleaned_response=cleaned_response,
                model_name=model_name,
                provider=provider,
                quality_score=quality_score,
                artifact_removal_score=artifact_score,
                source_dataset=source_dataset,
                preprocessing_steps=['basic_clean', 'artifact_removal'],
                response_length=len(cleaned_response),
                word_count=word_count,
                lexical_diversity=lexical_diversity,
                passes_filters=quality_score >= self.min_quality_score,
                timestamp=datetime.now().isoformat(),
                content_hash=content_hash
            )
            
            return signature
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        """Same cleaning as your original"""
        text = re.sub(r'<\|[^>]{0,100}?\|>', '', text)
        text = re.sub(r'\[INST\][\s\S]{0,1000}?\[/INST\]', '', text)
        text = re.sub(r'</?s>', '', text)
        text = re.sub(r'###.*?###', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _calculate_quality_score(self, text: str) -> float:
        """Same quality scoring as your original"""
        score = 1.0
        if self._has_repetition(text):
            score -= 0.3
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 5:
                score -= 0.2
        return max(0.0, min(1.0, score))

    def _calculate_artifact_removal_score(self, original: str, cleaned: str) -> float:
        """Same as your original"""
        if not original:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (len(original) - len(cleaned)) / len(original)))

    def _has_repetition(self, text: str) -> bool:
        """Same repetition check as your original"""
        words = text.split()
        if len(words) < 10:
            return False
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                return True
        return False

    def _clean_and_filter(self, signatures: List[ModelSignature]) -> List[ModelSignature]:
        """Basic filtering only - no deduplication"""
        filtered = []
        for sig in signatures:
            if sig.quality_score >= self.min_quality_score and sig.lexical_diversity >= 0.4:
                filtered.append(sig)
        return filtered

    def _generate_report(self, signatures: List[ModelSignature], collection_stats: Dict) -> Dict:
        """Same report format as your original"""
        model_counts = Counter(sig.model_name for sig in signatures)
        provider_counts = Counter(sig.provider for sig in signatures)
        dataset_counts = Counter(sig.source_dataset for sig in signatures)
        
        avg_quality = np.mean([sig.quality_score for sig in signatures]) if signatures else 0
        avg_length = np.mean([sig.response_length for sig in signatures]) if signatures else 0
        avg_diversity = np.mean([sig.lexical_diversity for sig in signatures]) if signatures else 0
        
        return {
            'total_samples': len(signatures),
            'model_distribution': dict(model_counts),
            'provider_distribution': dict(provider_counts),
            'dataset_distribution': dict(dataset_counts),
            'quality_metrics': {
                'average_quality_score': float(avg_quality),
                'average_response_length': float(avg_length),
                'average_lexical_diversity': float(avg_diversity)
            },
            'collection_timestamp': datetime.now().isoformat(),
            'source_type': 'enhanced_multi_provider_sources'
        }

    def _save_dataset(self, signatures: List[ModelSignature], report: Dict, output_path: str):
        """Same save format as your original"""
        data = {
            'signatures': [asdict(sig) for sig in signatures],
            'report': report
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Dataset saved to {output_path}")
            
        # Also save as CSV for easy inspection
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame([asdict(sig) for sig in signatures])
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved to {csv_path}")

def main():
    print("🚀 Collecting ENHANCED multi-provider sources with GATED datasets...")
    print("📊 TOP Non-OpenAI: LMSYS-Chat-1M, Chatbot Arena, OpenOrca")
    print("📊 GATED: Anthropic HH-RLHF, LMSYS-Chat-1M")
    print("📊 Plus: HC3, WildChat-1M, ShareGPT-GPT4, OpenAssistant")
    print("🔐 Using HuggingFace authentication for gated datasets")
    print("="*70)
    
    collector = NewSourcesCollector(
        target_per_model=2500,
        min_quality_score=0.5
    )
    
    result = collector.collect_new_signatures("enhanced_gated_sources_signatures.json")
    
    if result['total_samples'] > 0:
        print(f"\n🎯 ENHANCED COLLECTION complete: {result['total_samples']:,} samples")
        print(f"📁 Saved to: enhanced_gated_sources_signatures.json")
        
        # Print quick stats
        report = result['report']
        print(f"\n📊 Model Distribution:")
        for model, count in sorted(report['model_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count:,} samples")
            
        print(f"\n📊 Provider Distribution:")
        for provider, count in sorted(report['provider_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {provider}: {count:,} samples")
            
        print(f"\n📊 Dataset Distribution:")
        for dataset, count in sorted(report['dataset_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {dataset}: {count:,} samples")
    else:
        print("❌ No signatures collected")
        print("💡 Hint: If gated datasets failed, run: huggingface-cli login")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()