# Escalation Rules

Support agents should act autonomously when possible, but MUST escalate to a human in the following situations:

1. **High-Risk Actions**
   - Refunds over $50 require approval (or >$200 strict escalation).
   - Account deletion requests.
   - Any modifications to account billing details manually.

2. **Low Confidence**
   - If the agent's confidence in understanding the issue or the solution is low (e.g., < 0.70).
   - If no relevant knowledge base articles are found.

3. **Customer Sentiment**
   - The customer is very angry, using profanity, or highly frustrated.
   - The customer threatens legal action or media exposure.

4. **Unknown Intents**
   - The customer's request does not fall into standard categories (refund, password reset, billing faq).

**Escalation Process:**
When escalating, the agent must provide a summary of the situation, the customer's original message, the classified intent, retrieved policies, and the rationale for the escalation.
