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

class DataProcessingError(Exception):
    """Custom exception for dataset processing errors"""
    pass

@dataclass
class ModelSignature:
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

class EnhancedCollector:
    def __init__(self, target_per_model: int = 1500, min_quality_score: float = 0.6,
                 response_length_range: Tuple[int, int] = (80, 500),
                 min_lexical_diversity: float = 0.5, max_repetition_ratio: float = 0.3,
                 remove_non_ascii: bool = False):
        self.target_per_model = target_per_model
        self.min_quality_score = min_quality_score
        self.response_length_range = response_length_range
        self.min_lexical_diversity = min_lexical_diversity
        self.max_repetition_ratio = max_repetition_ratio
        self.remove_non_ascii = remove_non_ascii

        # Initialize content filters
        self.content_filters = [
            self._filter_placeholder_text,
        ]

        self.dataset_configs = {
            'daigt_v2': {
                'source': 'local_csv',
                'path': './data/daigt_v2/train_v2_drcat_02.csv',
                'model': 'various',
                'provider': 'various',
                'format': 'daigt_csv'
            },
            'daigt_v4_main': {
                'source': 'local_csv',
                'path': './data/daigt_v4/train_v4_drcat_01.csv',
                'model': 'various',
                'provider': 'various',
                'format': 'daigt_csv'
            },
            'daigt_v4_magic': {
                'source': 'local_csv',
                'path': './data/daigt_v4/daigt_magic_generations.csv',
                'model': 'various',
                'provider': 'various',
                'format': 'daigt_csv'       
            },
            'daigt_external': {
                'source': 'local_csv',
                'path': './data/daigt_external/daigt_external_dataset.csv',
                'model': 'various',
                'provider': 'various',
                'format': 'daigt_external'       
            },
            'human_detectors': {
                'source': 'local_json',
                'path': './data/human_detectors/human_detectors.json',
                'model': 'various',
                'provider': 'various',
                'format': 'expert_annotations'
            },
            'ultrachat_200k': {
                'dataset': 'HuggingFaceH4/ultrachat_200k',
                'split': 'train_sft',
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'format': 'conversation'
            },
            'wildchat': {
                'dataset': 'allenai/WildChat',
                'split': 'train',
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'format': 'conversation'
            },
            'wildchat_1m': {
                'dataset': 'allenai/WildChat-1M',
                'split': 'train',
                'model': 'various',
                'provider': 'openai',
                'format': 'wildchat'
            },
            'sharegpt': {
                'dataset': 'RyokoAI/ShareGPT52K',
                'split': 'train',
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'format': 'conversation'
            },
            'alpaca_cleaned': {
                'dataset': 'yahma/alpaca-cleaned',
                'split': 'train',
                'model': 'text-davinci-003',
                'provider': 'openai',
                'format': 'instruction'
            },
            'code_alpaca': {
                'dataset': 'sahil2801/CodeAlpaca-20k',
                'split': 'train',
                'model': 'code-davinci-002',
                'provider': 'openai',
                'format': 'instruction'
            },
            'anthropic_hh': {
                'dataset': 'Anthropic/hh-rlhf',
                'split': 'train',
                'model': 'claude-2',
                'provider': 'anthropic',
                'format': 'anthropic_hh'
            },
            'mage': {
                'dataset': 'yaful/MAGE',
                'split': 'train',
                'model': 'various',
                'provider': 'various',
                'format': 'detection'
            },
            'oasst2': {
                'dataset': 'OpenAssistant/oasst2',
                'split': 'train',
                'model': 'various',
                'provider': 'various',
                'format': 'conversation_tree'
            },
            'ultrafeedback': {
                'dataset': 'openbmb/UltraFeedback',
                'split': 'train',
                'model': 'various',
                'provider': 'various',
                'format': 'feedback'
            }
        }
        self.collected_hashes = set()
        self.model_patterns = [
            (re.compile(r'gpt-?4', re.I), 'gpt-4'),
            (re.compile(r'gpt-?3\.5|chatgpt', re.I), 'gpt-3.5-turbo'),
            (re.compile(r'claude-?3', re.I), 'claude-3'),
            (re.compile(r'claude-?2', re.I), 'claude-2'),
            (re.compile(r'claude(?!-[23])', re.I), 'claude'),
            (re.compile(r'llama-?2', re.I), 'llama-2'),
            (re.compile(r'llama(?!-2)', re.I), 'llama'),
            (re.compile(r'davinci', re.I), 'text-davinci-003'),
            (re.compile(r'palm|bard', re.I), 'palm'),
            (re.compile(r'gemini', re.I), 'gemini'),
            (re.compile(r'vicuna', re.I), 'vicuna'),
            (re.compile(r'alpaca', re.I), 'alpaca'),
        ]
        self._validate_configs()

    def _validate_configs(self):
        valid_configs = {}
        for name, config in self.dataset_configs.items():
            if config.get('source') in ['local_csv', 'local_json']:
                if not os.path.exists(config.get('path', '')):
                    logger.warning(f"Missing file for {name}: {config.get('path', 'unknown path')}")
                    continue
                valid_configs[name] = config
            else:  # HuggingFace datasets
                if not config.get('dataset'):
                    logger.warning(f"Missing dataset name for {name}")
                    continue
                valid_configs[name] = config
        self.dataset_configs = valid_configs

    def collect_enhanced_signatures(self, output_path: str = "enhanced_signatures.json") -> Dict[str, Any]:
        logger.info(f"Processing {len(self.dataset_configs)} datasets")
        all_signatures = []
        collection_stats = {}
        successful_datasets = 0
        dataset_items = list(self.dataset_configs.items())

        for i, (dataset_name, config) in enumerate(dataset_items, 1):
            logger.info(f"[{i}/{len(dataset_items)}] Processing: {dataset_name}")
            try:
                signatures = self._process_dataset(dataset_name, config)
                if signatures:
                    all_signatures.extend(signatures)
                    model_counts = Counter(sig.model_name for sig in signatures)
                    for model, count in model_counts.items():
                        collection_stats[model] = collection_stats.get(model, 0) + count
                    successful_datasets += 1
                    logger.info(f"✅ {dataset_name}: {len(signatures)} signatures")
            except Exception as e:
                logger.error(f"❌ {dataset_name}: {e}", exc_info=True)

        logger.info(f"Collection summary: {successful_datasets}/{len(self.dataset_configs)} successful")
        logger.info(f"Raw signatures: {len(all_signatures)}")

        if not all_signatures:
            return {'signatures': [], 'report': {}, 'total_samples': 0}

        deduplicated = self._remove_duplicates(all_signatures)
        logger.info(f"After deduplication: {len(deduplicated)}")

        cleaned = self._aggressive_cleaning(deduplicated)
        logger.info(f"After cleaning: {len(cleaned)}")

        filtered = self._content_filtering(cleaned)
        logger.info(f"After filtering: {len(filtered)}")

        final_signatures = self._final_quality_selection(filtered)
        logger.info(f"Final high-quality: {len(final_signatures)}")

        report = self._generate_report(final_signatures, collection_stats)
        self._save_dataset(final_signatures, report, output_path)

        return {
            'signatures': final_signatures,
            'report': report,
            'total_samples': len(final_signatures)
        }

    def _process_dataset(self, dataset_name: str, config: Dict) -> List[ModelSignature]:
        signatures = []
        source_type = config.get('source', 'huggingface')

        try:
            if source_type == 'local_csv':
                signatures = self._process_local_csv(dataset_name, config)
            elif source_type == 'local_json':
                signatures = self._process_local_json(dataset_name, config)
            else:
                signatures = self._process_huggingface(dataset_name, config)
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error in {dataset_name}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {dataset_name}: {e}")
            return []
        except FileNotFoundError as e:
            logger.error(f"File not found for {dataset_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in {dataset_name}: {e}", exc_info=True)
            return []

        return signatures

    def _process_local_csv(self, dataset_name: str, config: Dict) -> List[ModelSignature]:
        signatures = []
        chunksize = 10000
        max_rows = 80000  # Increased for DAIGT datasets
        processed = 0
        
        # Debug logging for DAIGT datasets
        if dataset_name.startswith('daigt'):
            logger.info(f"   Processing DAIGT dataset: {dataset_name}")

        for chunk in pd.read_csv(config['path'], chunksize=chunksize):
            if processed >= max_rows:
                break
                
            # Debug: log column names for first chunk
            if processed == 0 and dataset_name.startswith('daigt'):
                logger.info(f"   Available columns: {list(chunk.columns)}")

            for _, row in chunk.iterrows():
                if len(signatures) >= self.target_per_model * 3 or processed >= max_rows:  # Allow more per dataset
                    break

                try:
                    prompt, response = self._extract_content_local(row, config)
                    if prompt and response:  # Got content
                        if self._basic_quality_check(prompt, response):
                            model_name = self._extract_model_from_local_data(row, dataset_name)
                            provider = self._get_provider_from_model(model_name)
                            signature = self._create_signature(
                                prompt=prompt,
                                response=response,
                                model_name=model_name,  
                                provider=provider,
                                source_dataset=dataset_name,
                                index=processed
                            )
                            if signature:
                                signatures.append(signature)
                except (KeyError, ValueError, TypeError) as e:
                    continue
                processed += 1

            if processed % 10000 == 0:
                logger.info(f"   Processed {processed} rows, collected {len(signatures)} signatures")

        logger.info(f"   Final: {processed} rows processed, {len(signatures)} signatures collected")
        return signatures

    def _process_local_json(self, dataset_name: str, config: Dict) -> List[ModelSignature]:
        signatures = []
        with open(config['path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = list(data.values())
            
        logger.info(f"   Loaded JSON with {len(data)} items")

        for i, item in enumerate(data):
            if len(signatures) >= self.target_per_model:
                break

            try:
                prompt, response = self._extract_content_local(item, config)
                if self._basic_quality_check(prompt, response):
                    model_name = self._extract_model_from_local_data(item, dataset_name)
                    provider = self._get_provider_from_model(model_name)
                    signature = self._create_signature(
                        prompt=prompt,
                        response=response,
                        model_name=model_name,
                        provider=provider,
                        source_dataset=dataset_name,
                        index=i
                    )
                    if signature:
                        signatures.append(signature)
            except (KeyError, ValueError, TypeError):
                continue

        return signatures

    def _process_huggingface(self, dataset_name: str, config: Dict) -> List[ModelSignature]:
        signatures = []
        
        # Debug logging for problem datasets
        if dataset_name in ['mage']:
            logger.info(f"   🔍 Processing problem dataset: {dataset_name}")
        
        try:
            # Handle datasets with data_files
            load_kwargs = {
                'streaming': True,
                'trust_remote_code': config.get('trust_remote_code', False)
            }
            
            if 'data_files' in config:
                # For datasets that need specific files
                logger.info(f"   Loading {dataset_name} with data_files: {config['data_files']}")
                dataset = load_dataset(
                    config['dataset'], 
                    data_files=config['data_files'],
                    **load_kwargs
                )
                # Get the first available split
                if hasattr(dataset, 'keys'):
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]
                    logger.info(f"   Using split: {first_split}")
            else:
                # Standard loading with split
                split = config.get('split', 'train')
                dataset = load_dataset(
                    config['dataset'], 
                    split=split,
                    **load_kwargs
                )
                
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            # Try alternative loading methods
            try:
                dataset = load_dataset(
                    config['dataset'], streaming=True,
                    trust_remote_code=config.get('trust_remote_code', False)
                )
                # Use first available split
                if hasattr(dataset, 'keys'):
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]
            except Exception as e2:
                logger.error(f"Could not load {dataset_name} at all: {e2}")
                return []
        
        processed = 0
        max_items = 15000
        debug_logged = False

        for item in dataset:
            if len(signatures) >= self.target_per_model or processed >= max_items:
                break

            # Debug logging for first few items of problem datasets
            if dataset_name in ['mage'] and processed < 3 and not debug_logged:
                logger.info(f"   📋 Sample {dataset_name} item keys: {list(item.keys())}")
                if dataset_name == 'mage' and 'label' in item:
                    logger.info(f"   📋 MAGE label value: {item.get('label')} (0=machine, 1=human)")
                debug_logged = True

            try:
                prompt, response = self._extract_content(item, config)
                if self._basic_quality_check(prompt, response):
                    actual_model = config['model']
                    actual_provider = config['provider']

                    if config['model'] == 'various':
                        actual_model = self._extract_model_from_item(item, dataset_name)
                        actual_provider = self._get_provider_from_model(actual_model)

                    signature = self._create_signature(
                        prompt=prompt,
                        response=response,
                        model_name=actual_model,
                        provider=actual_provider,
                        source_dataset=dataset_name,
                        index=processed
                    )
                    if signature:
                        signatures.append(signature)
            except (KeyError, ValueError, TypeError):
                continue
            processed += 1

            if processed % 2000 == 0:
                logger.info(f"   Processed {processed} items, collected {len(signatures)} signatures")

        return signatures

    def _extract_content_local(self, item, config: Dict) -> Tuple[str, str]:
        format_type = config['format']
        if format_type == 'daigt_csv':
            if isinstance(item, pd.Series):
                # Check both label columns - prefer 'label' over 'RDizzl3_seven'
                is_ai_generated = False
                
                # Check 'label' column first (1 = AI-generated)
                if 'label' in item and item.get('label') == 1:
                    is_ai_generated = True
                # Fallback to 'RDizzl3_seven' column (True = AI-generated)  
                elif 'RDizzl3_seven' in item and item.get('RDizzl3_seven') == True:
                    is_ai_generated = True
                # Fallback to 'generated' column (1 = AI-generated)
                elif 'generated' in item and item.get('generated') == 1:
                    is_ai_generated = True
                
                if is_ai_generated:
                    # Extract prompt and text
                    prompt = str(item.get('prompt_name', item.get('source', 'Generate text.')))
                    text = str(item.get('text', ''))
                    
                    # Basic validation
                    if len(text) > 50 and len(text.split()) > 8:
                        return prompt, text
            return '', ''
        elif format_type == 'daigt_external':
            if isinstance(item, pd.Series):
                # For external dataset: instructions = prompt, source_text = AI response
                instruction = str(item.get('instructions', '')).strip()
                ai_response = str(item.get('source_text', '')).strip()
                
                # Clean up common task prefixes
                if instruction.startswith('\nTask:'):
                    instruction = instruction[6:].strip()
                
                # Ensure we have both components and reasonable length
                if instruction and ai_response and len(ai_response) > 50 and len(instruction) > 10:
                    return instruction, ai_response
            return '', ''
        elif format_type == 'expert_annotations':
            # More flexible matching for human detectors
            ground_truth = item.get('ground_truth', '').lower()
            if any(x in ground_truth for x in ['machine', 'ai', 'generated', 'artificial']):
                article = item.get('article', '') or item.get('text', '')
                if article and len(article) > 50:
                    return "Generate an article.", article
            return '', ''
        return '', ''

    def _extract_content(self, item, config: Dict) -> Tuple[str, str]:
        format_type = config['format']
        if format_type == 'conversation':
            return self._parse_conversation(item)
        elif format_type == 'instruction':
            prompt = item.get('instruction', '')
            response = item.get('output', '') or item.get('response', '')
            return prompt, response
        elif format_type == 'comparison':
            prompt = item.get('question', '')
            response = item.get('chatgpt_answers', [''])[0] if item.get('chatgpt_answers') else ''
            return prompt, response
        elif format_type == 'arena':
            conv = item.get('conversation_a', [])
            if len(conv) >= 2:
                return conv[0].get('content', ''), conv[1].get('content', '')
            return '', ''
        elif format_type == 'detection':
            # MAGE dataset - CRITICAL: label=0 means machine-generated, label=1 means human-written!
            prompt = item.get('prompt', '') or item.get('instruction', '') or item.get('input', '') or "Generate text."
            response = item.get('response', '') or item.get('text', '') or item.get('output', '')
            
            # Check for machine-generated content (label=0 in MAGE dataset)
            if item.get('label') == 0:  # 0 = machine-generated (what we want!)
                if response and len(response) > 50:
                    return prompt, response
            
            # Alternative: check generator field for non-human sources
            generator = item.get('generator', '').lower()
            if generator and 'human' not in generator and len(response) > 50:
                return prompt, response
                
            return '', ''
        elif format_type == 'conversation_tree':
            # OASST2 has role-based messages that alternate
            if item.get('role') == 'prompter' and item.get('parent_id'):
                # This is a user question, skip
                return '', ''
            elif item.get('role') == 'assistant':
                # This is an assistant response - find the parent prompter message
                prompt = item.get('text', '')
                response = ''
                # For now just use the text as response since we can't easily find parent
                if prompt:
                    return "Generate a helpful response.", prompt
            return '', ''
        elif format_type == 'feedback':
            prompt = item.get('instruction', '')
            responses = item.get('completions', [])
            if responses:
                return prompt, responses[0].get('response', '')
            return '', ''
        elif format_type == 'wildchat':
            conv = item.get('conversation', [])
            if len(conv) >= 2:
                return conv[0].get('content', ''), conv[1].get('content', '')
            return '', ''
        elif format_type == 'anthropic_hh':
            chosen = item.get('chosen', '')
            parts = chosen.split('\n\nAssistant:')
            if len(parts) >= 2:
                prompt = parts[0].replace('Human:', '').strip()
                response = parts[1].split('\n\nHuman:')[0].strip()
                return prompt, response
            return '', ''
        return '', ''

    def _parse_conversation(self, item) -> Tuple[str, str]:
        # Try different conversation structures
        messages = item.get('messages', []) or item.get('conversation', []) or item.get('conversations', [])
        
        if not messages or len(messages) < 2:
            return '', ''
            
        # Standard format with role-based messages
        if isinstance(messages[0], dict) and 'role' in messages[0]:
            prompt = ''
            response = ''
            for i, msg in enumerate(messages):
                if msg.get('role') in ['user', 'human'] and not prompt:
                    prompt = msg.get('content', '') or msg.get('value', '')
                elif msg.get('role') in ['assistant', 'gpt'] and prompt:
                    response = msg.get('content', '') or msg.get('value', '')
                    break
            return prompt, response
        
        # Simple alternating format
        elif len(messages) >= 2:
            prompt = messages[0] if isinstance(messages[0], str) else str(messages[0])
            response = messages[1] if isinstance(messages[1], str) else str(messages[1])
            return prompt, response
            
        return '', ''

    def _extract_model_from_item(self, item, dataset_name: str) -> str:
        if 'model' in item:
            return self._normalize_model_name(item['model'])
        elif 'model_name' in item:
            return self._normalize_model_name(item['model_name'])
        elif dataset_name == 'lmsys_arena':
            return self._normalize_model_name(item.get('model_a', 'unknown'))
        elif dataset_name == 'mage':
            return self._normalize_model_name(item.get('generator', 'unknown'))
        return 'unknown'

    def _extract_model_from_local_data(self, item, dataset_name: str) -> str:
        if dataset_name.startswith('daigt'):
            # Try 'model' column first (most direct)
            if hasattr(item, 'get') and 'model' in item:
                model = item.get('model')
                if model and model != 'human':
                    return self._normalize_model_name(model)
            
            # Try 'source' column for model info
            if hasattr(item, 'get') and 'source' in item:
                source = item.get('source', '')
                model = self._extract_model_from_source(source)
                if model != 'unknown':
                    return model
                    
        elif dataset_name == 'human_detectors':
            if hasattr(item, 'get'):
                return self._normalize_model_name(item.get('generation_model', ''))
        return 'unknown'

    def _extract_model_from_source(self, source: str) -> str:
        if not source:
            return 'unknown'
        source = source.lower()
        
        # Handle specific DAIGT source patterns
        if any(x in source for x in ['chat_gpt', 'chatgpt', 'gpt_moth']):
            return 'gpt-3.5-turbo'
        elif any(x in source for x in ['gpt-4', 'gpt4']):
            return 'gpt-4'
        elif any(x in source for x in ['mistral7b', 'mistral_7b', 'mistral-7b']):
            return 'mistral-7b'
        elif any(x in source for x in ['llama2', 'llama_2', 'llama-2']):
            return 'llama-2'
        elif any(x in source for x in ['llama_70b', 'llama-70b', 'llama70b']):
            return 'llama-70b'
        elif 'llama' in source:
            return 'llama'
        elif any(x in source for x in ['falcon_180b', 'falcon-180b', 'falcon180b']):
            return 'falcon-180b'
        elif 'falcon' in source:
            return 'falcon'
        elif any(x in source for x in ['claude', 'darragh_claude']):
            return 'claude-2'
        elif any(x in source for x in ['palm', 'bard', 'kingki19_palm']):
            return 'palm'
        elif any(x in source for x in ['gemini', 'bison']):
            return 'gemini'
        elif any(x in source for x in ['davinci', 'text-davinci']):
            return 'text-davinci-003'
        elif 'neural-chat' in source:
            return 'neural-chat-7b'
        elif any(x in source for x in ['bloom', 'bigscience']):
            return 'bloom'
        elif any(x in source for x in ['t5', 'flan']):
            return 't5'
        elif 'persuade_corpus' in source:
            return 'human'  # This indicates human-written content
        
        return 'unknown'

    def _normalize_model_name(self, model_name: str) -> str:
        if not model_name or model_name == 'unknown':
            return 'unknown'
        model_name = model_name.strip()
        for pattern, normalized_name in self.model_patterns:
            if pattern.search(model_name):
                return normalized_name
        return model_name.lower()

    def _get_provider_from_model(self, model_name: str) -> str:
        model_name = model_name.lower()
        if any(x in model_name for x in ['gpt', 'davinci', 'curie', 'chatgpt']):
            return 'openai'
        elif 'claude' in model_name:
            return 'anthropic'
        elif any(x in model_name for x in ['llama', 'vicuna', 'alpaca']):
            return 'meta'
        elif any(x in model_name for x in ['palm', 'bard', 'gemini']):
            return 'google'
        return 'unknown'

    def _basic_quality_check(self, prompt: str, response: str) -> bool:
        if not prompt or not response:
            return False
        
        # More lenient length check for DAIGT datasets
        min_length = 50  # Reduced minimum length
        max_length = 5000  # Increased maximum length
        if len(response) < min_length or len(response) > max_length:
            return False
            
        # More lenient word count
        if len(response.split()) < 5:  # Very permissive word count
            return False
            
        # Check for obvious artifacts but be more lenient
        severe_artifacts = ['<|endoftext|>', '<|startoftext|>']
        if any(artifact in response for artifact in severe_artifacts):
            return False
            
        return True

    def _create_signature(self, prompt: str, response: str, model_name: str,
                          provider: str, source_dataset: str, index: int) -> Optional[ModelSignature]:
        try:
            cleaned_response = self._clean_text(response)
            content_hash = hashlib.md5(cleaned_response.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
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
                passes_filters=quality_score >= self.min_quality_score and lexical_diversity >= self.min_lexical_diversity,
                timestamp=datetime.now().isoformat(),
                content_hash=content_hash
            )
            self.collected_hashes.add(content_hash)
            return signature
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        # Safer regex patterns with bounded repetitions
        text = re.sub(r'<\|[^>]{0,100}?\|>', '', text)  # Bounded match
        text = re.sub(r'\[INST\][\s\S]{0,1000}?\[/INST\]', '', text)  # Limit match length
        text = re.sub(r'</?s>', '', text)
        text = re.sub(r'###.*?###', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _calculate_quality_score(self, text: str) -> float:
        score = 1.0
        # Penalize repetition
        if self._has_repetition(text):
            score -= 0.3
        # Penalize short sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 5:
                score -= 0.2
        # Penalize excessive punctuation
        if text:
            punct_ratio = sum(1 for c in text if c in '!?.,;:') / len(text)
            if punct_ratio > 0.1:
                score -= 0.1
        return max(0.0, min(1.0, score))

    def _calculate_artifact_removal_score(self, original: str, cleaned: str) -> float:
        if not original:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (len(original) - len(cleaned)) / len(original)))

    def _has_repetition(self, text: str) -> bool:
        words = text.split()
        if len(words) < 10:
            return False
        # Fast check for consecutive repeated words
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                return True
        # Efficient n-gram repetition check
        max_ngram_size = min(5, len(words) // 2)
        for length in range(3, max_ngram_size + 1):
            if len(words) < length * 2:  # Need at least 2 instances
                continue
            phrase_counts = Counter()
            max_iterations = min(200, len(words) - length + 1)
            step = max(1, (len(words) - length + 1) // max_iterations) if max_iterations > 0 else 1
            for i in range(0, len(words) - length + 1, step):
                phrase = ' '.join(words[i:i+length])
                phrase_counts[phrase] += 1
            if phrase_counts:
                most_common_count = phrase_counts.most_common(1)[0][1]
                if most_common_count > 1 and most_common_count / len(phrase_counts) > self.max_repetition_ratio:
                    return True
        return False

    def _remove_duplicates(self, signatures: List[ModelSignature]) -> List[ModelSignature]:
        seen_hashes = set()
        unique_signatures = []
        for sig in signatures:
            if sig.content_hash not in seen_hashes:
                seen_hashes.add(sig.content_hash)
                unique_signatures.append(sig)
        return unique_signatures

    def _aggressive_cleaning(self, signatures: List[ModelSignature]) -> List[ModelSignature]:
        cleaned = []
        for sig in signatures:
            text = sig.cleaned_response
            # Remove URLs safely
            text = re.sub(r'https?://\S{1,100}', '', text)  # Limited length
            # Remove email addresses
            text = re.sub(r'\S{1,50}@\S{1,50}\.\S{1,10}', '', text)  # Bounded patterns
            # Remove excessive newlines
            text = re.sub(r'\n+', ' ', text)
            # Remove non-printable control characters
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            # Conditionally remove non-ASCII characters
            if self.remove_non_ascii:
                text = re.sub(r'[^\x00-\x7F]', '', text)
            text = text.strip()
            if len(text) >= self.response_length_range[0]:
                sig.cleaned_response = text
                sig.response_length = len(text)
                sig.word_count = len(text.split())
                cleaned.append(sig)
        return cleaned

    def _content_filtering(self, signatures: List[ModelSignature]) -> List[ModelSignature]:
        filtered = []
        for sig in signatures:
            text = sig.cleaned_response
            if all(filter_func(text) for filter_func in self.content_filters):
                # Update quality score after filtering
                sig.quality_score = self._calculate_quality_score(sig.cleaned_response)
                filtered.append(sig)
        return filtered

    # Content filter implementations
    def _filter_placeholder_text(self, text: str) -> bool:
        """Filter out responses with placeholder text"""
        placeholders = ['lorem ipsum', 'insert text', 'example text',
                        'placeholder', 'content here', 'type here']
        return not any(ph in text.lower() for ph in placeholders)

    def _final_quality_selection(self, signatures: List[ModelSignature]) -> List[ModelSignature]:
        # Filter by quality thresholds
        quality_filtered = [
            sig for sig in signatures
            if sig.quality_score >= self.min_quality_score
            and sig.lexical_diversity >= self.min_lexical_diversity
        ]
        # Balance by model
        balanced = self._balance_by_model(quality_filtered)
        return balanced

    def _balance_by_model(self, signatures: List[ModelSignature]) -> List[ModelSignature]:
        model_groups = defaultdict(list)
        for sig in signatures:
            model_groups[sig.model_name].append(sig)
        balanced = []
        for model, sigs in model_groups.items():
            # Sort by quality score and take the best
            sigs.sort(key=lambda x: x.quality_score, reverse=True)
            # Allow more samples from datasets with lots of data
            limit = self.target_per_model
            if len(sigs) > self.target_per_model * 3:  # If we have 3x more than target
                limit = min(self.target_per_model * 3, len(sigs))  # Allow up to 3x target
            elif len(sigs) > self.target_per_model * 2:  # If we have 2x more than target
                limit = min(self.target_per_model * 2, len(sigs))  # Allow up to 2x target
            balanced.extend(sigs[:limit])
        return balanced

    def _generate_report(self, signatures: List[ModelSignature], collection_stats: Dict) -> Dict:
        model_counts = Counter(sig.model_name for sig in signatures)
        provider_counts = Counter(sig.provider for sig in signatures)
        dataset_counts = Counter(sig.source_dataset for sig in signatures)
        avg_quality = np.mean([sig.quality_score for sig in signatures]) if signatures else 0
        avg_length = np.mean([sig.response_length for sig in signatures]) if signatures else 0
        avg_diversity = np.mean([sig.lexical_diversity for sig in signatures]) if signatures else 0
        # Generate sample outputs for validation
        samples_by_model = {}
        for model in list(model_counts.keys())[:5]:  # Top 5 models
            model_sigs = [sig for sig in signatures if sig.model_name == model]
            if model_sigs:
                best_sample = max(model_sigs, key=lambda x: x.quality_score)
                samples_by_model[model] = {
                    'prompt': best_sample.prompt[:200] + "..." if len(best_sample.prompt) > 200 else best_sample.prompt,
                    'response': best_sample.response[:300] + "..." if len(best_sample.response) > 300 else best_sample.response,
                    'quality_score': best_sample.quality_score,
                    'lexical_diversity': best_sample.lexical_diversity
                }
        # Class balance analysis (assuming AI-generated text is the positive class)
        total_samples = len(signatures)
        balance_ratio = 1.0  # All samples are AI-generated in this collector
        # Quality distribution analysis
        quality_scores = [sig.quality_score for sig in signatures] if signatures else []
        quality_distribution = {
            'min': float(np.min(quality_scores)) if quality_scores else 0,
            'max': float(np.max(quality_scores)) if quality_scores else 0,
            'median': float(np.median(quality_scores)) if quality_scores else 0,
            'std': float(np.std(quality_scores)) if quality_scores else 0
        }
        return {
            'total_samples': total_samples,
            'model_distribution': dict(model_counts),
            'provider_distribution': dict(provider_counts),
            'dataset_distribution': dict(dataset_counts),
            'quality_metrics': {
                'average_quality_score': float(avg_quality),
                'average_response_length': float(avg_length),
                'average_lexical_diversity': float(avg_diversity),
                'quality_distribution': quality_distribution
            },
            'validation_stats': {
                'class_balance': {
                    'ai_generated_samples': total_samples,
                    'human_samples': 0,  # This collector focuses on AI text
                    'balance_ratio': balance_ratio
                },
                'dataset_coverage': len(dataset_counts),
                'model_coverage': len(model_counts),
                'samples_per_model': dict(model_counts)
            },
            'sample_outputs': samples_by_model,
            'collection_timestamp': datetime.now().isoformat(),
            'raw_collection_stats': collection_stats
        }

    def _save_dataset(self, signatures: List[ModelSignature], report: Dict, output_path: str):
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
 
        # Save configuration for reproducibility
        config_path = output_path.replace('.json', '_config.json')
        self._save_config(config_path)

    def _save_config(self, config_path: str):
        """Save configuration parameters for reproducibility"""
        config_data = {
            'collection_parameters': {
                'target_per_model': self.target_per_model,
                'min_quality_score': self.min_quality_score,
                'response_length_range': self.response_length_range,
                'min_lexical_diversity': self.min_lexical_diversity,
                'max_repetition_ratio': self.max_repetition_ratio,
                'remove_non_ascii': self.remove_non_ascii
            },
            'dataset_configs': {name: {k: v for k, v in config.items() if k != 'path'}
                               for name, config in self.dataset_configs.items()},
            'collection_timestamp': datetime.now().isoformat(),
            'version': '1.1'
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {config_path}")

def print_validation_report(result: Dict[str, Any]):
    """Print a comprehensive validation report"""
    if not result.get('report'):
        return
    report = result['report']
    print("\n" + "="*60)
    print("📊 VALIDATION REPORT")
    print("="*60)
    # Basic stats
    print(f"Total Samples: {report['total_samples']:,}")
    print(f"Datasets Used: {report['validation_stats']['dataset_coverage']}")
    print(f"Models Covered: {report['validation_stats']['model_coverage']}")
    # Quality metrics
    quality = report['quality_metrics']
    print(f"\nQuality Metrics:")
    print(f"  Average Quality Score: {quality['average_quality_score']:.3f}")
    print(f"  Average Response Length: {quality['average_response_length']:.1f} chars")
    print(f"  Average Lexical Diversity: {quality['average_lexical_diversity']:.3f}")
    quality_dist = quality['quality_distribution']
    print(f"  Quality Score Range: {quality_dist['min']:.3f} - {quality_dist['max']:.3f}")
    print(f"  Quality Score Std Dev: {quality_dist['std']:.3f}")
    # Model distribution
    print(f"\nModel Distribution:")
    for model, count in sorted(report['model_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / report['total_samples']) * 100
        print(f"  {model}: {count:,} samples ({percentage:.1f}%)")
    # Dataset distribution
    print(f"\nDataset Distribution:")
    for dataset, count in sorted(report['dataset_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / report['total_samples']) * 100
        print(f"  {dataset}: {count:,} samples ({percentage:.1f}%)")
    # Sample outputs
    if 'sample_outputs' in report and report['sample_outputs']:
        print(f"\nSample Outputs (Top Quality):")
        for model, sample in list(report['sample_outputs'].items())[:3]:
            print(f"\n  {model} (Quality: {sample['quality_score']:.3f}):")
            print(f"    Prompt: {sample['prompt']}")
            print(f"    Response: {sample['response']}")
    print("="*60)

def main():
    print("🚀 Starting dataset collection...")
    print("="*50)
    collector = EnhancedCollector(
        target_per_model=3000,  # Keep high target
        min_quality_score=0.5,  # Keep permissive
        response_length_range=(50, 5000),  # Keep expanded range
        min_lexical_diversity=0.4,  # Keep permissive
        max_repetition_ratio=0.4,  # Keep permissive
        remove_non_ascii=False
    )
    result = collector.collect_enhanced_signatures("enhanced_signatures.json")
    if result['total_samples'] > 0:
        print(f"\n🎯 Collection complete: {result['total_samples']:,} samples")
        if result['total_samples'] > 45000:
            print("🌟 LEGENDARY! Over 45K samples - you've built the ultimate AI detection dataset!")
        elif result['total_samples'] > 40000:
            print("🌟 INCREDIBLE SUCCESS! You've built a world-class AI detection dataset!")
        print(f"📁 Data saved to: enhanced_signatures.json")
        print(f"⚙️  Config saved to: enhanced_signatures_config.json")
        print_validation_report(result)
    else:
        print("❌ No signatures collected - check dataset availability")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()