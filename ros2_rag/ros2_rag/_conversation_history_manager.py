from typing import Dict, Tuple


class ConversationHistoryManager:

    def __init__(self, recent_interaction_number: int,
                 evicted_interaction_number: int):
        # Copy vars
        self._recent_interaction_number = recent_interaction_number
        self._evicted_interaction_number = evicted_interaction_number

        # Summary
        self._summary = {}
        self._summary['user_goals'] = []
        self._summary['constraints'] = []
        self._summary['context'] = []

        # Recent interactions
        self._recent_interations = []
        # Evicted interactions
        self._evicted_interations = []

    def store_new_interaction(self, query: str, completion: str):
        # Last interaction
        self._recent_interations.insert(
            0, {'query': query, 'completion': completion})

        # Pass recent interaction to evicted if required
        if len(self._recent_interations) > self._recent_interaction_number:
            # Oldest to evicted
            self._evicted_interations.insert(0, self._recent_interations[-1])
            # Remove from recent
            self._recent_interations.pop()

        # Manage evicted interaction number
        if len(self._evicted_interations) > self._evicted_interaction_number:
            self._evicted_interations.pop()

    def _divide_goal_constraints_context(
            self, summary_completion: str) -> Tuple[bool, list, list, list]:
        # Define lists
        goals = []
        constraints = []
        context = []

        # Get completion lines
        lines = (
            [line.strip()
             for line in summary_completion.splitlines() if line.strip()]
        )

        # Divide sections
        sections = {'tmp': []}
        current_section = 'tmp'

        for line in lines:
            if 'goals' in line.lower() and ':' in line:
                current_section = 'user_goals'
                sections[current_section] = []
            elif 'constraint' in line.lower() and ':' in line:
                current_section = 'constraints'
                sections[current_section] = []
            elif 'context' in line.lower() and ':' in line:
                current_section = 'context'
                sections[current_section] = []
            else:
                sections[current_section].append(line.lstrip('-').strip())

        # Asign lists if not empty
        if 'user_goals' in sections:
            goals = sections['user_goals']
        if 'constraints' in sections:
            constraints = sections['constraints']
        if 'context' in sections:
            context = sections['context']

        return True, goals, constraints, context

    def store_summary(self, summary_completion: str) -> bool:
        # Divide completion in goal/constraints/context
        res, goals, constraints, context = (
            self._divide_goal_constraints_context(summary_completion))

        if res:
            # Store data
            self._summary['user_goals'] = goals
            self._summary['constraints'] = constraints
            self._summary['context'] = context

            return True
        else:
            return False

    def get_evicted_interactions(self) -> Tuple[bool, list, list]:
        # Check if evicted_interactions stored
        if len(self._evicted_interations) == 0:
            return False, [], []

        # Get queries & completion
        queries = []
        completions = []

        for interactions in self._evicted_interations:
            queries.append(interactions['query'])
            completions.append(interactions['completion'])

        return True, queries, completions

    def get_summary(self) -> Dict:
        return self._summary.copy()

    def get_history_prompt(self) -> str:
        # Check if any interaction stored
        if len(self._recent_interations) == 0:
            return ''

        # Header
        prompt = 'SYSTEM:\nYou are an assistant helping the user ' \
            'achieve their goals.\n'
        # Formatting instructions
        prompt += 'Respond in the most natural format for the content. ' \
            'Do not default to numbered lists unless they improve clarity.\n\n'

        # Check if evicted interactions & summary available
        if len(self._evicted_interations) > 0:
            prompt += 'User Memory (Summary of prior interactions):\n'

            # Insert summary - User goals
            prompt += '- Goals: '
            for goal in self._summary['user_goals']:
                prompt += goal.rstrip('.') + '.'
            prompt += '\n'
            # Insert summary - Constraints
            prompt += '- Constraints: '
            for contraint in self._summary['constraints']:
                prompt += contraint.rstrip('.') + '.'
            prompt += '\n'
            # Insert summary - Context
            prompt += '- Context: '
            for context in self._summary['context']:
                prompt += context.rstrip('.') + '.'
            prompt += '\n\n'

        # Insert last interactions
        prompt += 'Conversation so far:\n'
        for interaction in reversed(self._recent_interations):
            prompt += 'User: ' + interaction['query'] + '\n'
            prompt += 'Assistant: ' + interaction['completion'] + '\n'
        prompt += '\n\n'

        # Final
        prompt += 'User: '

        return prompt

    def get_conversation_summary_as_string(self) -> str:
        # Check if evicted interactions & summary available
        if len(self._evicted_interations) == 0:
            return ''

        # Insert user goals
        user_goals = 'User goals: '
        for goal in self._summary['user_goals']:
            user_goals += goal.rstrip('.') + '. '
        user_goals += '\n'
        # Insert constriants
        constraints = 'Constraints: '
        for contraint in self._summary['constraints']:
            constraints += contraint.rstrip('.') + '. '
        constraints += '\n'
        # Insert context
        complete_context = 'Context: '
        for context in self._summary['context']:
            complete_context += context.rstrip('.') + '. '
        complete_context += '\n'

        return user_goals + constraints + complete_context

    def get_last_user_queries_as_string(self) -> str:
        # Check if any interaction stored
        if len(self._recent_interations) == 0:
            return ''

        prompt = ''

        for interaction in reversed(self._recent_interations):
            prompt += 'User: ' + interaction['query'] + '\n'

        return prompt
