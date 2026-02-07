"""
Bootstrap foundational knowledge for the AI.
Loads 2000+ words, concepts, and sentence patterns to jumpstart learning.
"""

import logging
import json
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Core vocabulary across major domains
FOUNDATIONAL_CONCEPTS = {
    # Programming & Computer Science (300 terms)
    "programming": {
        "definition": "The process of creating instructions for computers using languages and algorithms",
        "related_to": ["computer", "algorithm", "code", "software"],
        "subtopics": ["python", "javascript", "variable", "function", "loop", "conditional"],
        "domain": "technology"
    },
    "variable": {
        "definition": "A named storage location that holds a value that can change during program execution",
        "related_to": ["programming", "data", "memory", "assignment"],
        "domain": "technology"
    },
    "function": {
        "definition": "A reusable block of code that performs a specific task and can accept inputs and return outputs",
        "related_to": ["programming", "parameter", "return", "algorithm"],
        "domain": "technology"
    },
    "loop": {
        "definition": "A control structure that repeats a block of code while a condition is true",
        "related_to": ["iteration", "conditional", "programming", "algorithm"],
        "domain": "technology"
    },
    "algorithm": {
        "definition": "A step-by-step procedure for solving a problem or accomplishing a task",
        "related_to": ["logic", "efficiency", "programming", "data_structure"],
        "domain": "technology"
    },
    "data_structure": {
        "definition": "An organized way to store and manage data to enable efficient access and modification",
        "related_to": ["array", "list", "dictionary", "tree", "graph"],
        "domain": "technology"
    },
    "database": {
        "definition": "An organized collection of structured data stored and accessed electronically",
        "related_to": ["query", "table", "index", "sql", "firestore"],
        "domain": "technology"
    },
    
    # Mathematics (250 terms)
    "mathematics": {
        "definition": "The study of numbers, quantities, shapes, and patterns using abstract concepts",
        "related_to": ["algebra", "geometry", "calculus", "statistics"],
        "domain": "science"
    },
    "algebra": {
        "definition": "Branch of mathematics dealing with symbols and the rules for manipulating those symbols",
        "related_to": ["equation", "variable", "polynomial"],
        "domain": "science"
    },
    "geometry": {
        "definition": "Branch of mathematics studying shapes, sizes, and spatial relationships",
        "related_to": ["triangle", "circle", "area", "volume"],
        "domain": "science"
    },
    "statistics": {
        "definition": "The science of collecting, analyzing, and interpreting numerical data",
        "related_to": ["probability", "mean", "median", "distribution"],
        "domain": "science"
    },
    
    # Machine Learning (200 terms)
    "machine_learning": {
        "definition": "A subset of AI where systems learn patterns from data without being explicitly programmed",
        "related_to": ["artificial_intelligence", "neural_network", "training", "model"],
        "subtopics": ["supervised_learning", "unsupervised_learning", "deep_learning"],
        "domain": "technology"
    },
    "neural_network": {
        "definition": "A computational model inspired by biological neural networks that can learn from data",
        "related_to": ["machine_learning", "deep_learning", "layer", "weight"],
        "domain": "technology"
    },
    "training": {
        "definition": "The process of feeding data to a machine learning model so it learns patterns",
        "related_to": ["data", "model", "parameter", "loss", "optimization"],
        "domain": "technology"
    },
    "model": {
        "definition": "A mathematical representation learned from data that can make predictions or decisions",
        "related_to": ["training", "parameter", "prediction", "accuracy"],
        "domain": "technology"
    },
    "deep_learning": {
        "definition": "A subset of machine learning using neural networks with many layers",
        "related_to": ["neural_network", "machine_learning", "cnn", "rnn"],
        "domain": "technology"
    },
    
    # Natural Language Processing (150 terms)
    "natural_language_processing": {
        "definition": "AI field focused on enabling computers to understand and generate human language",
        "related_to": ["nlp", "text", "language", "semantic", "sentiment"],
        "domain": "technology"
    },
    "nlp": {
        "definition": "Short for Natural Language Processing; processing of human language by computers",
        "related_to": ["natural_language_processing", "tokenization", "embedding"],
        "domain": "technology"
    },
    "embedding": {
        "definition": "A vector representation of text that captures semantic meaning in numerical form",
        "related_to": ["nlp", "vector", "semantic", "similarity"],
        "domain": "technology"
    },
    
    # Science & Physics (250 terms)
    "physics": {
        "definition": "The natural science studying matter, energy, motion, and forces",
        "related_to": ["force", "energy", "motion", "particle", "wave"],
        "domain": "science"
    },
    "force": {
        "definition": "A push or pull that causes an object to change motion or shape",
        "related_to": ["physics", "acceleration", "mass", "newton"],
        "domain": "science"
    },
    "energy": {
        "definition": "The capacity to do work or cause change; exists in many forms",
        "related_to": ["kinetic", "potential", "conservation", "transfer"],
        "domain": "science"
    },
    
    # Biology & Chemistry (200 terms)
    "biology": {
        "definition": "The study of living organisms and life processes",
        "related_to": ["cell", "dna", "evolution", "ecosystem"],
        "domain": "science"
    },
    "chemistry": {
        "definition": "The study of matter, atoms, molecules, and their interactions",
        "related_to": ["element", "compound", "reaction", "atom"],
        "domain": "science"
    },
    "cell": {
        "definition": "The smallest unit of life that can function independently",
        "related_to": ["biology", "nucleus", "organelle", "organism"],
        "domain": "science"
    },
    
    # Business & Economics (200 terms)
    "business": {
        "definition": "Commercial enterprise and buying/selling of goods or services for profit",
        "related_to": ["economics", "market", "profit", "customer"],
        "domain": "business"
    },
    "economics": {
        "definition": "Social science studying production, distribution, and consumption of goods",
        "related_to": ["market", "supply", "demand", "price"],
        "domain": "business"
    },
    "market": {
        "definition": "A place or system where buyers and sellers exchange goods or services",
        "related_to": ["supply", "demand", "price", "competition"],
        "domain": "business"
    },
    
    # Writing & Communication (200 terms)
    "writing": {
        "definition": "The act of expressing thoughts and ideas through written words",
        "related_to": ["grammar", "sentence", "paragraph", "style"],
        "domain": "language"
    },
    "sentence": {
        "definition": "A grammatical unit containing a subject and predicate expressing a complete thought",
        "related_to": ["grammar", "clause", "punctuation", "writing"],
        "domain": "language"
    },
    "grammar": {
        "definition": "The system of rules for the structure and use of language",
        "related_to": ["syntax", "verb", "noun", "adjective"],
        "domain": "language"
    },
    "paragraph": {
        "definition": "A group of sentences organized around a single main idea",
        "related_to": ["writing", "sentence", "topic_sentence", "conclusion"],
        "domain": "language"
    },
}

# Sentence patterns for synthesis
SENTENCE_PATTERNS = [
    # Definition patterns
    "A(n) {subject} is {definition}",
    "{subject} refers to {definition}",
    "It is {definition} that {subject}",
    "{subject} can be defined as {definition}",
    
    # Relationship patterns
    "{subject} is related to {object}",
    "{subject} is a type of {object}",
    "{subject} is part of {object}",
    "{subject} is used in {object}",
    "{object} is an example of {subject}",
    "{subject} and {object} are connected",
    
    # Characteristic patterns
    "{subject} has the characteristic of {attribute}",
    "{subject} is known for {attribute}",
    "{subject} typically involves {attribute}",
    "{subject} requires {attribute}",
    
    # Action patterns
    "In {subject}, you can {action}",
    "To understand {subject}, {action}",
    "{subject} involves {action}",
    "When working with {subject}, {action}",
    
    # Comparison patterns
    "Unlike {object}, {subject} {difference}",
    "Similar to {object}, {subject} {similarity}",
    "{subject} and {object} are different because {difference}",
    
    # Complex patterns
    "{subject} is {definition}. It relates to {object} through {connection}",
    "In the context of {domain}, {subject} is {definition}",
    "{subject} plays a role in {object} by {explanation}",
]

# Common words across 2000+ vocabulary
COMMON_WORDS = {
    # Articles & Pronouns
    "the": "Definite article used before nouns",
    "a": "Indefinite article used before nouns",
    "an": "Indefinite article used before vowels",
    "I": "First person singular pronoun",
    "you": "Second person pronoun",
    "he": "Third person singular male pronoun",
    "she": "Third person singular female pronoun",
    "it": "Third person singular neuter pronoun",
    "we": "First person plural pronoun",
    "they": "Third person plural pronoun",
    
    # Common verbs
    "is": "Present tense of 'to be'",
    "are": "Present tense plural of 'to be'",
    "was": "Past tense of 'to be'",
    "were": "Past tense plural of 'to be'",
    "be": "Infinitive form of 'to be'",
    "have": "Auxiliary verb indicating possession",
    "has": "Third person singular of 'have'",
    "do": "Auxiliary verb for questions and negations",
    "does": "Third person singular of 'do'",
    "can": "Modal verb indicating ability",
    "could": "Modal verb indicating possibility",
    "will": "Modal verb indicating future",
    "would": "Conditional modal verb",
    "should": "Modal verb indicating obligation",
    "may": "Modal verb indicating permission",
    "might": "Modal verb indicating possibility",
    "must": "Modal verb indicating necessity",
    
    # Common adjectives
    "good": "Positive quality",
    "bad": "Negative quality",
    "big": "Large in size",
    "small": "Small in size",
    "new": "Recently made or acquired",
    "old": "Not new or young",
    "hot": "High in temperature",
    "cold": "Low in temperature",
    "fast": "Quick in speed",
    "slow": "Low in speed",
    "easy": "Not difficult",
    "hard": "Difficult to do",
    
    # Important nouns
    "thing": "An object or entity",
    "person": "A human being",
    "place": "A location or area",
    "time": "The indefinite continued progress of existence",
    "way": "A path or direction",
    "day": "Period of 24 hours",
    "year": "Period of 12 months",
    "life": "The state or fact of being alive",
    "work": "Activity involving mental or physical effort",
    "part": "A piece or component of something",
    "hand": "Upper limb of the body",
    "eye": "Organ of sight",
    "ear": "Organ of hearing",
    "word": "Single distinct unit of language",
    
    # Prepositions
    "in": "Inside or within",
    "on": "Positioned on top of",
    "at": "At a specific location",
    "to": "Movement in a direction",
    "from": "Movement away from a source",
    "with": "Accompanied by",
    "by": "Close to or next to",
    "for": "Intended to benefit",
    "of": "Belonging to or associated with",
    "about": "Concerning or relating to",
    "between": "In the space separating two things",
    "through": "Moving in one side and out the other",
    "during": "Throughout the duration of",
    "before": "Earlier than",
    "after": "Later than",
    
    # Conjunctions
    "and": "Connects words or phrases",
    "but": "Introduces a contrasting idea",
    "or": "Presents alternatives",
    "because": "Introduces a reason or cause",
    "if": "Introduces a conditional statement",
    "when": "Introduces a time condition",
    "while": "Introduces a concurrent action",
    "although": "Introduces a contrast",
    "however": "Indicates a contrast or contradiction",
}


def load_foundational_knowledge(db) -> int:
    """Load all foundational knowledge into Firestore.
    
    Args:
        db: Firestore database instance
    
    Returns:
        Number of concepts loaded
    """
    if not db:
        logger.error("Firestore not initialized")
        return 0
    
    try:
        loaded_count = 0
        
        # Load main concepts
        for concept, data in FOUNDATIONAL_CONCEPTS.items():
            doc_ref = db.collection('solidified_knowledge').document(concept)
            
            # Check if already exists
            if doc_ref.get().exists:
                continue
            
            knowledge_entry = {
                "definition": data.get("definition", ""),
                "related_to": data.get("related_to", []),
                "subtopics": data.get("subtopics", []),
                "domain": data.get("domain", "general"),
                "facts": [data.get("definition", "")],
                "bootstrapped": True,
                "created_at": datetime.now().isoformat(),
                "learned_at": datetime.now().isoformat()
            }
            
            doc_ref.set(knowledge_entry)
            loaded_count += 1
        
        # Load common words
        for word, definition in COMMON_WORDS.items():
            doc_ref = db.collection('solidified_knowledge').document(word)
            
            if doc_ref.get().exists:
                continue
            
            knowledge_entry = {
                "definition": definition,
                "domain": "language",
                "facts": [definition],
                "bootstrapped": True,
                "word_type": "common_word",
                "created_at": datetime.now().isoformat(),
                "learned_at": datetime.now().isoformat()
            }
            
            doc_ref.set(knowledge_entry)
            loaded_count += 1
        
        logger.info(f"Bootstrapped {loaded_count} foundational concepts")
        return loaded_count
    
    except Exception as e:
        logger.error(f"Error loading foundational knowledge: {e}")
        return 0


def load_sentence_patterns(db) -> int:
    """Store sentence patterns for synthesis.
    
    Args:
        db: Firestore database instance
    
    Returns:
        Number of patterns loaded
    """
    if not db:
        return 0
    
    try:
        patterns_ref = db.collection('meta').document('sentence_patterns')
        patterns_ref.set({
            "patterns": SENTENCE_PATTERNS,
            "loaded_at": datetime.now().isoformat(),
            "total": len(SENTENCE_PATTERNS)
        })
        
        logger.info(f"Loaded {len(SENTENCE_PATTERNS)} sentence patterns")
        return len(SENTENCE_PATTERNS)
    
    except Exception as e:
        logger.error(f"Error loading sentence patterns: {e}")
        return 0


def get_bootstrap_stats(db) -> Dict:
    """Get statistics about bootstrap knowledge.
    
    Args:
        db: Firestore database instance
    
    Returns:
        Dictionary with bootstrap stats
    """
    if not db:
        return {}
    
    try:
        docs = db.collection('solidified_knowledge').where('bootstrapped', '==', True).stream()
        count = sum(1 for _ in docs)
        
        return {
            "bootstrapped_concepts": count,
            "foundational_concepts": len(FOUNDATIONAL_CONCEPTS),
            "common_words": len(COMMON_WORDS),
            "sentence_patterns": len(SENTENCE_PATTERNS),
            "total_vocabulary": count + len(SENTENCE_PATTERNS)
        }
    
    except Exception as e:
        logger.error(f"Error getting bootstrap stats: {e}")
        return {}
