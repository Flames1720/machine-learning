import logging
from nltk.corpus import wordnet
from brain.db import db

logger = logging.getLogger(__name__)

def find_and_map_relationships(word):
    """
    Finds and maps relationships for a given word using WordNet.
    """
    logger.info(f"Finding relationships for: {word}")
    
    # Get the synsets for the word
    synsets = wordnet.synsets(word)
    
    if not synsets:
        logger.info(f"No WordNet synsets found for '{word}'.")
        return

    # Use the first synset as the primary one
    main_synset = synsets[0]

    # Get hypernyms (more general concepts)
    hypernyms = main_synset.hypernyms()
    hypernym_names = [h.lemmas()[0].name() for h in hypernyms]

    # Get hyponyms (more specific concepts)
    hyponyms = main_synset.hyponyms()
    hyponym_names = [h.lemmas()[0].name() for h in hyponyms]

    # Get meronyms (parts of)
    part_meronyms = main_synset.part_meronyms()
    part_meronym_names = [m.lemmas()[0].name() for m in part_meronyms]

    # Get holonyms (wholes that this is a part of)
    member_holonyms = main_synset.member_holonyms()
    member_holonym_names = [h.lemmas()[0].name() for h in member_holonyms]

    # Update Firestore with the new relationships
    doc_ref = db.collection('solidified_knowledge').document(word)
    doc_ref.update({
        'related_to': list(set(hypernym_names + member_holonym_names)),
        'subtopics': list(set(hyponym_names + part_meronym_names))
    })

    logger.info(f"Mapped relationships for '{word}':")
    logger.info(f"  - Hypernyms (related_to): {hypernym_names}")
    logger.info(f"  - Hyponyms (subtopics): {hyponym_names}")
    logger.info(f"  - Meronyms (subtopics): {part_meronym_names}")
    logger.info(f"  - Holonyms (related_to): {member_holonym_names}")
