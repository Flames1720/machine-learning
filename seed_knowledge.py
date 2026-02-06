
import time
from brain.knowledge_base import upsert_knowledge

def main():
    """Seeds the knowledge base with initial data."""
    print("--- Seeding Knowledge Base ---")

    knowledge_data = [
        {
            "id": "quantum-computing-advancements",
            "text": "Recent advancements in quantum computing include the development of more stable qubits, the creation of new quantum algorithms, and the construction of larger, more powerful quantum processors. Companies like Google, IBM, and Rigetti are making significant strides in building fault-tolerant quantum computers.",
            "metadata": {"topic": "Quantum Computing"}
        },
        {
            "id": "general-relativity-explanation",
            "text": "General relativity is Einstein's theory of gravity. In simple terms, it describes gravity not as a force, but as a curvature of spacetime caused by mass and energy. Imagine a bowling ball on a trampoline; the ball creates a dip in the fabric, and other objects roll towards it. That's a simplified analogy for how gravity works in general relativity.",
            "metadata": {"topic": "Physics"}
        },
        {
            "id": "learning-programming-languages",
            "text": "The best way to learn a new programming language is through a combination of structured learning and hands-on practice. Start with the fundamentals like syntax, data types, and control structures. Then, build small projects to apply what you've learned. Consistent practice and working on real-world problems are key to mastery.",
            "metadata": {"topic": "Programming"}
        }
    ]

    for item in knowledge_data:
        print(f"Upserting knowledge for: {item['id']}")
        upsert_knowledge(id=item['id'], text=item['text'], metadata=item['metadata'])
        time.sleep(1) # simple delay to avoid overwhelming any services

    print("\n--- Knowledge Base Seeding Complete ---")

if __name__ == "__main__":
    main()
