import ollama

system = '''
You are a first grade teacher. Give the easy a mark from 1 to 5. 1 being very bad and 5 being very good. Be concise, output only be score nothing else
'''

# promt = '''
# Which animal is faster: sharks or dolphins?
# 1. The fastest shark breed is the shortfin mako shark, which can reach speeds around 74 km/h.
# 2. The fastest dolphin breed is the common dolphin, which can reach speeds around 60 km/h.
# '''


# promt = '''
# Which animal is faster: cats or dogs? Follow these steps to find an answer:
# 1. Determine the speed of the fastest dog breed.
# 2. Determine the speed of the fastest cat breed.
# 3. Determine which one is faster.
# '''

promt = '''
Help me write a concise prompt for an application that grades college essays between 1 and 5
'''

#
# ollama.create(model='example',
#               from_='llama3.2',
#               parameters={
#                   "temperature": 0.1,
#               },
#               system=system
#               )

response = ollama.generate(model='llama3.2', prompt=promt)
print(response["response"])
