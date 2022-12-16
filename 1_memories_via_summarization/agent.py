import openai

class Agent():
  def __init__(self, name):
    self.name = name
    self.other_memory = {}
    self.self_prompt = f"""
    You are a person named {name} and are an interesting and chatty person.
    """
    self.conversation_so_far = ""

  def get_completion(self, prompt, stop = None):
      response = openai.Completion.create(
        temperature=0.8,
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        stop=stop

      )
      return response.choices[0].text

  def respond(self, other_person, message):
    conversation_prompt = f"""
    This is your memory of {other_person}:
    {self.other_memory.get(other_person,'This is the first time you are interacting.')}

    This is your current conversation with {other_person}:
    {self.conversation_so_far}
    """

    conversation = f"""
    {other_person}: {message}
    You: """

    prompt = self.self_prompt + conversation_prompt + conversation

    response = self.get_completion(prompt, [f'{other_person}:'])
    self.conversation_so_far += conversation + response

    print(f'----\n{self.conversation_so_far}\n----')

  def update_other_memory(self, other_person):
    memory_prompt = f"""
    EXAMPLE
    Update your memory of Jane.
    
    Existing memory:
    - Jane I are friends. We went to the same high school and college. We both like to play tennis.
    New lines of conversation:
    Jane: Hey! How was your holiday?
    Me: It was a lot of fun! I got to see my parents and my sister. How about you?
    Jane: It was great! I went to visit my parents in Upstate New York. We went ice skating and hiking with my family.
    Updated memory:
    - Jane and I are friends. We went to the same high school and college. We both like to play tennis.
    - Her parents live in Upstate New York. She went ice skating and hiking with their family over the holidays.
    END OF EXAMPLE

    Update your memory of {other_person}.

    Existing memory:
    {self.other_memory.get(other_person, '')}
    New lines of conversation:
    {self.conversation_so_far}
    Updated memory:
    """

    response = self.get_completion(memory_prompt)

    self.other_memory[other_person] = response