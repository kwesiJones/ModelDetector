import random
from enum import Enum
from typing import List, Dict

class ComplexityLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"  
    ADVANCED = "advanced"

class TopicDomain(Enum):
    TECHNICAL = "technical"
    CREATIVE = "creative"
    REASONING = "reasoning"
    BUSINESS = "business"
    EDUCATIONAL = "educational"

class PromptGenerator:
    """High-quality curated prompts for AI model differentiation"""
    
    def __init__(self):
        self.prompts = {
            TopicDomain.TECHNICAL.value: {
                ComplexityLevel.BASIC.value: [
                    "Write a Python function to reverse a string",
                    "Explain what an API is in simple terms",
                    "How do you create a variable in JavaScript?",
                    "What is the difference between HTTP and HTTPS?",
                    "Write HTML code for a basic webpage with a heading and paragraph"
                ],
                ComplexityLevel.INTERMEDIATE.value: [
                    "Implement a binary search algorithm in Python with error handling",
                    "Explain the difference between SQL joins with examples",
                    "Create a React component that manages form state",
                    "Design a REST API for a simple blog application",
                    "Write a function to validate email addresses using regex"
                ],
                ComplexityLevel.ADVANCED.value: [
                    "Implement a distributed caching system architecture",
                    "Optimize this database query for performance at scale",
                    "Design a microservices architecture for an e-commerce platform",
                    "Explain the CAP theorem and its implications for distributed systems",
                    "Implement a custom neural network layer in PyTorch"
                ]
            },
            TopicDomain.CREATIVE.value: {
                ComplexityLevel.BASIC.value: [
                    "Write a short story about a talking cat",
                    "Create a poem about the ocean",
                    "Describe a magical forest in detail",
                    "Write dialogue between two friends meeting after years",
                    "Create a recipe for an imaginary dish"
                ],
                ComplexityLevel.INTERMEDIATE.value: [
                    "Write a mystery story with an unexpected twist ending",
                    "Create a compelling product description for a new gadget",
                    "Write a persuasive email to convince someone to try a new restaurant",
                    "Develop character backstories for a fantasy novel",
                    "Write a movie script scene with tension and conflict"
                ],
                ComplexityLevel.ADVANCED.value: [
                    "Write a complex narrative with multiple timelines and perspectives",
                    "Create a comprehensive world-building guide for a sci-fi universe",
                    "Develop a brand story that emotionally connects with millennials",
                    "Write a philosophical dialogue exploring the nature of consciousness",
                    "Create an interactive story structure with branching narratives"
                ]
            },
            TopicDomain.REASONING.value: {
                ComplexityLevel.BASIC.value: [
                    "If all birds can fly and penguins are birds, why can't penguins fly?",
                    "A farmer has 17 sheep. All but 9 die. How many are left?",
                    "What comes next in this sequence: 2, 4, 8, 16, ?",
                    "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
                    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?"
                ],
                ComplexityLevel.INTERMEDIATE.value: [
                    "Three friends split a $30 bill. They each pay $10, but get $5 change. Where did the missing dollar go?",
                    "You're in a room with two doors. One leads to certain death, one to freedom. Each door has a guard - one always lies, one always tells truth. What single question can you ask?",
                    "A company's profits increased by 20% then decreased by 20%. What's the net change?",
                    "Explain why correlation does not imply causation with real examples",
                    "You have 8 balls, one is heavier. Using a balance scale only twice, how do you find the heavy ball?"
                ],
                ComplexityLevel.ADVANCED.value: [
                    "Analyze the logical fallacies in this argument and propose a stronger version",
                    "Given incomplete information, what assumptions would you make and why?",
                    "Design an experiment to test whether people make better decisions alone or in groups",
                    "Explain the prisoner's dilemma and its applications in business strategy",
                    "How would you approach solving a problem with multiple conflicting stakeholder interests?"
                ]
            },
            TopicDomain.BUSINESS.value: {
                ComplexityLevel.BASIC.value: [
                    "Write a professional email declining a meeting request",
                    "Explain the concept of supply and demand",
                    "What are the main components of a business plan?",
                    "How do you calculate profit margin?",
                    "List 5 ways to improve customer service"
                ],
                ComplexityLevel.INTERMEDIATE.value: [
                    "Develop a go-to-market strategy for a new SaaS product",
                    "Analyze the pros and cons of remote work for productivity",
                    "Create a competitive analysis framework for startups",
                    "Design a customer retention strategy for an e-commerce business",
                    "Explain different pricing strategies and when to use each"
                ],
                ComplexityLevel.ADVANCED.value: [
                    "Develop a comprehensive digital transformation strategy for a traditional retailer",
                    "Design a risk management framework for international expansion",
                    "Create a data-driven approach to optimize marketing spend across channels",
                    "Analyze the strategic implications of emerging AI technologies on business models",
                    "Develop a sustainable competitive advantage framework for a mature industry"
                ]
            },
            TopicDomain.EDUCATIONAL.value: {
                ComplexityLevel.BASIC.value: [
                    "Explain photosynthesis in simple terms",
                    "What are the main causes of World War I?",
                    "How do you solve this equation: 2x + 5 = 15?",
                    "What is DNA and why is it important?",
                    "Explain the water cycle with examples"
                ],
                ComplexityLevel.INTERMEDIATE.value: [
                    "Compare and contrast democracy and authoritarianism",
                    "Explain how compound interest works with practical examples",
                    "Describe the process of cellular respiration and its importance",
                    "Analyze the themes in Shakespeare's Romeo and Juliet",
                    "Explain climate change causes and potential solutions"
                ],
                ComplexityLevel.ADVANCED.value: [
                    "Critically analyze the effectiveness of different economic theories in modern contexts",
                    "Examine the ethical implications of genetic engineering in humans",
                    "Evaluate the impact of social media on democratic processes",
                    "Analyze the relationship between language and thought in cognitive science",
                    "Assess the role of artificial intelligence in shaping future education systems"
                ]
            }
        }
    
    def generate_prompts(self, domain: TopicDomain, complexity: ComplexityLevel, count: int = 5) -> List[str]:
        """Generate high-quality prompts for AI model testing"""
        available_prompts = self.prompts.get(domain.value, {}).get(complexity.value, [])
        
        if not available_prompts:
            return []
        
        # Return random sample, or repeat if we need more than available
        if count <= len(available_prompts):
            return random.sample(available_prompts, count)
        else:
            # Repeat prompts if we need more than available
            repeated = available_prompts * (count // len(available_prompts) + 1)
            return repeated[:count]
    
    def generate_mixed_prompts(self, count: int, domain: TopicDomain = None) -> List[str]:
        """Generate prompts across different complexity levels"""
        if domain is None:
            domains = list(TopicDomain)
            domain = random.choice(domains)
        
        complexities = list(ComplexityLevel)
        prompts = []
        
        per_complexity = count // len(complexities)
        remaining = count % len(complexities)
        
        for i, complexity in enumerate(complexities):
            batch_size = per_complexity + (1 if i < remaining else 0)
            batch_prompts = self.generate_prompts(domain, complexity, batch_size)
            prompts.extend(batch_prompts)
        
        random.shuffle(prompts)
        return prompts
    
    def generate_diverse_prompts(self, count: int) -> List[str]:
        """Generate prompts across all domains and complexities"""
        domains = list(TopicDomain)
        complexities = list(ComplexityLevel) 
        prompts = []
        
        for _ in range(count):
            domain = random.choice(domains)
            complexity = random.choice(complexities)
            batch = self.generate_prompts(domain, complexity, 1)
            if batch:
                prompts.extend(batch)
        
        return prompts[:count]
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return [domain.value for domain in TopicDomain]
    
    def get_available_complexities(self) -> List[str]:
        """Get list of available complexity levels"""  
        return [complexity.value for complexity in ComplexityLevel]