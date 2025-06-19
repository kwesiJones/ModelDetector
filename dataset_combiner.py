#!/usr/bin/env python3
"""
Enhanced Dataset Combiner
- Fuzzy duplicate detection with MinHash
- Language filtering 
- Schema validation
- Improved versioning
"""

import json
import logging
import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pydantic import BaseModel, ValidationError, Field

# Optional imports with fallbacks
try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    print("⚠️  datasketch not available. Install with: pip install datasketch")
    MINHASH_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("⚠️  langdetect not available. Install with: pip install langdetect")
    LANGDETECT_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Schema validation models
class SignatureSchema(BaseModel):
    """Pydantic schema for signature validation"""
    id: str
    prompt: str
    response: str
    cleaned_response: str
    model_name: str
    provider: str
    quality_score: float = Field(ge=0.0, le=1.0)
    response_length: int = Field(gt=0)
    word_count: int = Field(gt=0)
    lexical_diversity: float = Field(ge=0.0, le=1.0)
    content_hash: str
    language: Optional[str] = "en"
    fuzzy_hash: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields

@dataclass
class CombinerStats:
    """Statistics tracking for the combination process"""
    clean_input: int = 0
    messy_input: int = 0  
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    language_filtered: int = 0
    quality_filtered: int = 0
    validation_failed: int = 0
    final_output: int = 0

class EnhancedDatasetCombiner:
    """Enhanced combiner with fuzzy dedup, language filtering, and validation"""
    
    def __init__(self, 
                 target_per_model: int = 2500,
                 min_quality_score: float = 0.6,
                 enable_fuzzy_dedup: bool = True,
                 fuzzy_threshold: float = 0.8,
                 enable_language_filter: bool = True,
                 target_language: str = "en",
                 enable_validation: bool = True):
        
        self.target_per_model = target_per_model
        self.min_quality_score = min_quality_score
        self.enable_fuzzy_dedup = enable_fuzzy_dedup and MINHASH_AVAILABLE
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_language_filter = enable_language_filter and LANGDETECT_AVAILABLE
        self.target_language = target_language
        self.enable_validation = enable_validation
        
        # Tracking sets
        self.exact_hashes: Set[str] = set()
        self.fuzzy_lsh: Optional[MinHashLSH] = None
        self.stats = CombinerStats()
        
        # Initialize fuzzy deduplication
        if self.enable_fuzzy_dedup:
            self.fuzzy_lsh = MinHashLSH(threshold=fuzzy_threshold, num_perm=128)
            logger.info(f"✅ Fuzzy deduplication enabled (threshold: {fuzzy_threshold})")
        
        # Log feature availability
        if self.enable_language_filter:
            logger.info(f"✅ Language filtering enabled (target: {target_language})")
        if self.enable_validation:
            logger.info("✅ Schema validation enabled")
    
    def combine_datasets(self,
                        clean_dataset_path: str,
                        messy_dataset_path: str, 
                        output_path: str = "enhanced_training_dataset.json") -> Dict:
        """Enhanced combination with all improvements"""
        
        logger.info("🚀 Starting enhanced dataset combination...")
        logger.info(f"🔍 Active features:")
        logger.info(f"   - Fuzzy deduplication: {'✅' if self.enable_fuzzy_dedup else '❌'}")
        logger.info(f"   - Language filtering: {'✅' if self.enable_language_filter else '❌'}")  
        logger.info(f"   - Schema validation: {'✅' if self.enable_validation else '❌'}")
        logger.info(f"   - Enhanced quality scoring: ✅")
        logger.info(f"   - Model balancing: ✅")
        logger.info("")
        
        # Load datasets
        clean_signatures = self._load_dataset(clean_dataset_path, "clean")
        messy_signatures = self._load_dataset(messy_dataset_path, "messy")
        
        self.stats.clean_input = len(clean_signatures)
        self.stats.messy_input = len(messy_signatures)
        
        if not clean_signatures and not messy_signatures:
            logger.error("No data to combine!")
            return {}
        
        logger.info("🔄 Processing signatures...")
        
        # Process clean signatures (minimal processing)
        processed_clean = self._process_signatures(clean_signatures, is_clean=True)
        
        # Process messy signatures (full cleaning)
        processed_messy = self._process_signatures(messy_signatures, is_clean=False)
        
        # Combine and deduplicate
        all_signatures = processed_clean + processed_messy
        deduplicated = self._advanced_deduplication(all_signatures)
        
        # Final quality filtering and balancing
        final_signatures = self._final_processing(deduplicated)
        
        self.stats.final_output = len(final_signatures)
        
        # Generate report and save
        report = self._generate_enhanced_report(final_signatures)
        result = self._save_enhanced_dataset(final_signatures, report, output_path)
        
        # Print summary
        self._print_enhanced_summary()
        
        return result
    
    def _load_dataset(self, path: str, dataset_type: str) -> List[Dict]:
        """Load dataset with error handling"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f) 
            signatures = data.get('signatures', [])
            logger.info(f"✅ Loaded {dataset_type} dataset: {len(signatures):,} signatures")
            return signatures
        except FileNotFoundError:
            logger.warning(f"⚠️  {dataset_type} dataset not found: {path}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading {dataset_type} dataset: {e}")
            return []
    
    def _process_signatures(self, signatures: List[Dict], is_clean: bool) -> List[Dict]:
        """Process signatures with optional deep cleaning and progress tracking"""
        import time
        
        processed = []
        total = len(signatures)
        start_time = time.time()
        
        logger.info(f"Processing {total:,} {'clean' if is_clean else 'messy'} signatures...")
        if not is_clean and total > 10000:
            estimated_minutes = (total / 1000) * 0.5  # Rough estimate: 0.5 min per 1K signatures
            logger.info(f"⏱️  Estimated processing time: {estimated_minutes:.1f} minutes")
        
        for i, sig in enumerate(signatures):
            # Progress tracking for large datasets
            if i > 0 and i % 2500 == 0:
                progress_pct = (i/total)*100
                elapsed = time.time() - start_time
                if i > 0:
                    estimated_total = (elapsed / i) * total
                    remaining = max(0, estimated_total - elapsed)
                    logger.info(f"   {'Clean' if is_clean else 'Messy'}: {i:,}/{total:,} ({progress_pct:.1f}%) - {len(processed):,} valid - ETA: {remaining/60:.1f}min")
                else:
                    logger.info(f"   {'Clean' if is_clean else 'Messy'}: {i:,}/{total:,} ({progress_pct:.1f}%) - {len(processed):,} valid so far")
            
            try:
                # Apply cleaning if needed
                if not is_clean:
                    sig = self._deep_clean_signature(sig)
                    if not sig:
                        self.stats.quality_filtered += 1
                        continue
                
                # Add language detection (with sampling for speed)
                if self.enable_language_filter:
                    if not self._check_language(sig):
                        self.stats.language_filtered += 1
                        continue
                
                # Add fuzzy hash
                if self.enable_fuzzy_dedup:
                    sig = self._add_fuzzy_hash(sig)
                
                # Validate schema
                if self.enable_validation:
                    if not self._validate_signature(sig):
                        self.stats.validation_failed += 1
                        continue
                
                processed.append(sig)
                
            except KeyboardInterrupt:
                logger.warning("Processing interrupted by user")
                break
            except Exception as e:
                logger.debug(f"Failed to process signature {i}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(processed):,}/{total:,} signatures")
        return processed
    
    def _deep_clean_signature(self, signature: Dict) -> Optional[Dict]:
        """Enhanced deep cleaning"""
        try:
            response = signature.get('response', '') or signature.get('cleaned_response', '')
            if not response:
                return None
            
            # Deep cleaning steps
            cleaned = response
            
            # Remove artifacts and special tokens
            cleaned = re.sub(r'<\|[^>]*\|>', '', cleaned)
            cleaned = re.sub(r'\[INST\].*?\[/INST\]', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'###.*?###', '', cleaned)
            cleaned = re.sub(r'</?s>', '', cleaned)
            cleaned = re.sub(r'Human:|Assistant:', '', cleaned)
            
            # Remove URLs, emails, and phone numbers
            cleaned = re.sub(r'https?://\S+', '', cleaned)
            cleaned = re.sub(r'\S+@\S+\.\S+', '', cleaned) 
            cleaned = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', cleaned)
            
            # Remove excessive punctuation and emojis
            cleaned = re.sub(r'[!?]{3,}', '!', cleaned)
            cleaned = re.sub(r'\.{3,}', '...', cleaned)
            cleaned = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', '', cleaned)
            
            # Clean whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
            
            # Enhanced quality checks
            if not self._enhanced_quality_check(cleaned):
                return None
            
            # Update signature
            signature = signature.copy()
            signature.update({
                'cleaned_response': cleaned,
                'response_length': len(cleaned),
                'word_count': len(cleaned.split()),
                'lexical_diversity': self._calculate_lexical_diversity(cleaned),
                'quality_score': self._calculate_enhanced_quality_score(cleaned),
                'content_hash': hashlib.md5(cleaned.encode()).hexdigest(),
                'preprocessing_steps': signature.get('preprocessing_steps', []) + ['enhanced_deep_clean']
            })
            
            return signature
            
        except Exception:
            return None
    
    def _enhanced_quality_check(self, text: str) -> bool:
        """More sophisticated quality checking"""
        if not text or len(text) < 30:
            return False
        
        words = text.split()
        if len(words) < 8:
            return False
        
        # Check for excessive repetition
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:  # Less than 30% unique words
                return False
        
        # Check for reasonable sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return False
        
        # Check for excessive capitalization
        if sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            return False
        
        return True
    
    def _check_language(self, signature: Dict) -> bool:
        """Check if text is in target language with timeout and sampling"""
        if not self.enable_language_filter:
            return True
            
        try:
            text = signature.get('cleaned_response', signature.get('response', ''))
            if len(text) < 50:  # Too short for reliable detection
                signature['language'] = 'unknown'
                return True
            
            # Sample text for faster detection (first 500 chars)
            sample_text = text[:500] if len(text) > 500 else text
            
            # Quick detection with timeout simulation
            detected_lang = detect(sample_text)
            signature['language'] = detected_lang
            return detected_lang == self.target_language
            
        except (LangDetectException, Exception) as e:
            # If detection fails or times out, assume it's valid
            logger.debug(f"Language detection failed: {e}")
            signature['language'] = 'unknown'
            return True
    
    def _add_fuzzy_hash(self, signature: Dict) -> Dict:
        """Add MinHash for fuzzy duplicate detection"""
        if not self.enable_fuzzy_dedup:
            return signature
            
        try:
            text = signature.get('cleaned_response', '')
            words = text.lower().split()
            
            # Create MinHash
            mh = MinHash(num_perm=128)
            for word in words:
                mh.update(word.encode('utf-8'))
            
            # Store as hex string
            signature['fuzzy_hash'] = mh.digest().hex()
            
        except Exception:
            signature['fuzzy_hash'] = None
            
        return signature
    
    def _validate_signature(self, signature: Dict) -> bool:
        """Validate signature against schema"""
        if not self.enable_validation:
            return True
            
        try:
            # Remove fields that might cause validation issues
            # Use model_fields for Pydantic v2 compatibility
            valid_fields = getattr(SignatureSchema, 'model_fields', getattr(SignatureSchema, '__fields__', {}))
            clean_sig = {k: v for k, v in signature.items() 
                        if k in valid_fields or k in ['preprocessing_steps', 'timestamp', 'source_dataset']}
            
            SignatureSchema(**clean_sig)
            return True
        except ValidationError as e:
            logger.debug(f"Validation failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False
    
    def _advanced_deduplication(self, signatures: List[Dict]) -> List[Dict]:
        """Advanced deduplication with exact and fuzzy matching"""
        deduplicated = []
        
        # Track fuzzy duplicates with LSH
        if self.enable_fuzzy_dedup and self.fuzzy_lsh:
            fuzzy_hashes = {}
        
        for sig in signatures:
            content_hash = sig.get('content_hash', '')
            
            # Check exact duplicates
            if content_hash in self.exact_hashes:
                self.stats.exact_duplicates_removed += 1
                continue
            
            # Check fuzzy duplicates
            if self.enable_fuzzy_dedup and sig.get('fuzzy_hash'):
                try:
                    fuzzy_hash = sig['fuzzy_hash']
                    mh = MinHash(num_perm=128)
                    mh.digest(bytes.fromhex(fuzzy_hash))
                    
                    # Check if similar document exists
                    similar = self.fuzzy_lsh.query(mh)
                    if similar:
                        self.stats.fuzzy_duplicates_removed += 1
                        continue
                    
                    # Add to LSH
                    self.fuzzy_lsh.insert(content_hash, mh)
                    
                except Exception:
                    pass  # Continue with exact dedup only
            
            # Add to exact hash set
            self.exact_hashes.add(content_hash)
            deduplicated.append(sig)
        
        logger.info(f"Deduplication: {len(signatures)} → {len(deduplicated)} signatures")
        return deduplicated
    
    def _final_processing(self, signatures: List[Dict]) -> List[Dict]:
        """Final quality filtering and model balancing"""
        # Quality filtering
        quality_filtered = [
            sig for sig in signatures 
            if sig.get('quality_score', 0) >= self.min_quality_score
        ]
        
        # Model balancing
        model_groups = defaultdict(list)
        for sig in quality_filtered:
            model_groups[sig.get('model_name', 'unknown')].append(sig)
        
        balanced = []
        for model, sigs in model_groups.items():
            # Sort by quality
            sigs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Take top N per model
            limit = min(self.target_per_model, len(sigs))
            balanced.extend(sigs[:limit])
        
        return balanced
    
    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (TTR - Type Token Ratio)"""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _calculate_enhanced_quality_score(self, text: str) -> float:
        """Enhanced quality scoring"""
        score = 1.0
        
        # Length score
        if len(text) < 100:
            score -= 0.2
        elif len(text) > 2000:
            score -= 0.1
        
        # Lexical diversity
        diversity = self._calculate_lexical_diversity(text)
        if diversity < 0.3:
            score -= 0.3
        elif diversity > 0.7:
            score += 0.1
        
        # Sentence structure
        sentences = text.split('.')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3:
                score -= 0.2
            elif 8 <= avg_sentence_length <= 25:
                score += 0.1
        
        # Repetition penalty
        words = text.split()
        if len(words) > 10:
            consecutive_repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
            if consecutive_repeats > len(words) * 0.1:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_enhanced_report(self, signatures: List[Dict]) -> Dict:
        """Generate comprehensive report with enhancement stats"""
        model_counts = Counter(sig.get('model_name', 'unknown') for sig in signatures)
        language_counts = Counter(sig.get('language', 'unknown') for sig in signatures)
        
        quality_scores = [sig.get('quality_score', 0) for sig in signatures]
        
        return {
            'dataset_info': {
                'total_samples': len(signatures),
                'version': '2.0',
                'creation_timestamp': datetime.now().isoformat(),
                'ready_for_training': True
            },
            'enhancement_features': {
                'fuzzy_deduplication': self.enable_fuzzy_dedup,
                'language_filtering': self.enable_language_filter,
                'schema_validation': self.enable_validation,
                'target_language': self.target_language if self.enable_language_filter else None
            },
            'processing_stats': asdict(self.stats),
            'quality_metrics': {
                'average_quality_score': float(np.mean(quality_scores)) if quality_scores else 0,
                'min_quality_score': float(np.min(quality_scores)) if quality_scores else 0,
                'max_quality_score': float(np.max(quality_scores)) if quality_scores else 0,
                'quality_std': float(np.std(quality_scores)) if quality_scores else 0
            },
            'distributions': {
                'models': dict(model_counts),
                'languages': dict(language_counts)
            }
        }
    
    def _save_enhanced_dataset(self, signatures: List[Dict], report: Dict, output_path: str) -> Dict:
        """Save with both JSON and CSV formats"""
        result = {
            'signatures': signatures,
            'report': report,
            'total_samples': len(signatures)
        }
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save CSV
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame(signatures)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"💾 Enhanced dataset saved: {output_path}")
        logger.info(f"💾 Training CSV saved: {csv_path}")
        
        return result
    
    def _print_enhanced_summary(self):
        """Print detailed summary of enhancements"""
        print("\n" + "="*70)
        print("🎯 ENHANCED DATASET COMBINATION COMPLETE")
        print("="*70)
        
        print(f"📊 Input Processing:")
        print(f"   Clean Dataset: {self.stats.clean_input:,} samples")
        print(f"   Messy Dataset: {self.stats.messy_input:,} samples")
        print(f"   Total Input: {self.stats.clean_input + self.stats.messy_input:,} samples")
        
        print(f"\n🔍 Quality Filtering:")
        print(f"   Quality Filtered: {self.stats.quality_filtered:,}")
        print(f"   Language Filtered: {self.stats.language_filtered:,}")
        print(f"   Validation Failed: {self.stats.validation_failed:,}")
        
        print(f"\n🔧 Deduplication:")
        print(f"   Exact Duplicates Removed: {self.stats.exact_duplicates_removed:,}")
        if self.enable_fuzzy_dedup:
            print(f"   Fuzzy Duplicates Removed: {self.stats.fuzzy_duplicates_removed:,}")
        
        print(f"\n✅ Final Output:")
        print(f"   Training-Ready Samples: {self.stats.final_output:,}")
        
        # Calculate efficiency
        total_removed = (self.stats.quality_filtered + self.stats.language_filtered + 
                        self.stats.validation_failed + self.stats.exact_duplicates_removed + 
                        self.stats.fuzzy_duplicates_removed)
        efficiency = (self.stats.final_output / (self.stats.clean_input + self.stats.messy_input)) * 100 if (self.stats.clean_input + self.stats.messy_input) > 0 else 0
        
        print(f"\n📈 Processing Efficiency: {efficiency:.1f}%")
        print(f"   Removed: {total_removed:,} low-quality/duplicate samples")
        
        print("="*70)


def main():
    """Enhanced main function"""
    print("🚀 ENHANCED DATASET COMBINER v2.0")
    print("Features: Fuzzy Dedup + Language Filter + Schema Validation")
    print("="*60)
    
    # Check dependencies
    missing_deps = []
    if not MINHASH_AVAILABLE:
        missing_deps.append("datasketch")
    if not LANGDETECT_AVAILABLE:
        missing_deps.append("langdetect")
    
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        print("Continuing with available features...\n")
    
    # Full-featured mode - all checks enabled
    print("🔬 Using FULL PROCESSING MODE:")
    print("   - Language detection: ENABLED")
    print("   - Fuzzy deduplication: ENABLED (if available)")
    print("   - Schema validation: ENABLED")
    print("   - Enhanced quality scoring: ENABLED")
    print("   - Progress tracking: ENABLED")
    print("   ⏱️  This will take longer but produce the highest quality dataset")
    print()
    
    # Configure enhanced combiner with ALL features
    combiner = EnhancedDatasetCombiner(
        target_per_model=2500,
        min_quality_score=0.6,
        enable_fuzzy_dedup=MINHASH_AVAILABLE,  # Enable if available
        fuzzy_threshold=0.8,
        enable_language_filter=LANGDETECT_AVAILABLE,  # Enable if available
        target_language="en",
        enable_validation=True
    )
    
    # File paths (CORRECTED)
    CLEAN_DATASET = "enhanced_signatures.json"
    MESSY_DATASET = "enhanced_gated_sources_signatures.json"  # Corrected filename!
    OUTPUT_DATASET = "enhanced_training_dataset.json"
    
    # Run enhanced combination
    result = combiner.combine_datasets(
        clean_dataset_path=CLEAN_DATASET,
        messy_dataset_path=MESSY_DATASET,
        output_path=OUTPUT_DATASET
    )
    
    if result:
        print(f"\n🎯 Enhanced training dataset ready:")
        print(f"   📄 JSON: {OUTPUT_DATASET}")
        print(f"   📊 CSV: {OUTPUT_DATASET.replace('.json', '.csv')}")
        print(f"   🔍 Total samples: {result['total_samples']:,}")
        
        if missing_deps or not combiner.enable_language_filter:
            print(f"\n💡 ENHANCEMENT STATUS:")
            if missing_deps:
                print(f"   Missing: {' '.join(missing_deps)} - install with: pip install {' '.join(missing_deps)}")
            if combiner.enable_language_filter:
                print(f"   Language filtering: ✅ ENABLED")
            else:
                print(f"   Language filtering: ❌ DISABLED (langdetect not available)")
            if combiner.enable_fuzzy_dedup:
                print(f"   Fuzzy deduplication: ✅ ENABLED")
            else:
                print(f"   Fuzzy deduplication: ❌ DISABLED (datasketch not available)")
            print(f"   Schema validation: ✅ ENABLED")
    else:
        print("❌ Enhanced combination failed")

if __name__ == "__main__":
    main()