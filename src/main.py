import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import CONFIG

# Ensure the GEMINI_API_KEY is set
if not CONFIG.get("GEMINI_API_KEY"):
    print("❌ ERROR: GEMINI_API_KEY is not set.")
    print("   Please copy .env.example to .env and set your Google Gemini API key:")
    print("   cp .env.example .env")
    sys.exit(1)

from src.perception import PerceptionModule
from src.memory import MemoryModule
from src.reasoning import ReasoningModule
from src.action import ActionModule

def process_ticket(message: str, perception: PerceptionModule, memory: MemoryModule, reasoning: ReasoningModule, action: ActionModule):
    print(f"\n{'='*50}")
    print(f"📩 New Ticket: \"{message}\"")
    print(f"{'='*50}")
    
    try:
        # 1. Perception
        intent = perception.parse_message(message)
        print(f"   [Parsed Intent] {intent.intent} (Urgency: {intent.urgency}, Sentiment: {intent.sentiment})")
        
        # 2. Memory (Retrieval)
        # We always retrieve escalation rules in addition to the intent-specific query
        retrieval_query = f"{intent.intent} policy and rules"
        policies = memory.retrieve(retrieval_query, n_results=2)
        
        # 3. Reasoning
        decision = reasoning.decide_action(intent, policies)
        
        # 4. Action
        action.execute(message, intent, decision)
        
    except Exception as e:
        print(f"❌ An error occurred processing the ticket: {e}")

def main():
    print("🤖 Initializing Support Router Agent...")
    
    perception = PerceptionModule()
    memory = MemoryModule()
    reasoning = ReasoningModule()
    action = ActionModule()
    
    print("\n✅ Agent Ready! Processing test cases...\n")
    
    test_cases = [
        "Hi, I forgot my password for john.doe@example.com. Can you help?",
        "I need a refund for my last order #9999. It was $30 and I bought it yesterday.",
        "I demand a refund immediately! Order #8888 was $150 and your service is terrible!",
        "Can you refund order #7777? It was $300 and I don't need it anymore.",
        "How do I update my credit card info?",
        "Delete my account and all my data right now."
    ]
    
    for case in test_cases:
        process_ticket(case, perception, memory, reasoning, action)
        print("\nPress Enter to process the next ticket...")
        # input() # Uncomment to step through manually

if __name__ == "__main__":
    main()
