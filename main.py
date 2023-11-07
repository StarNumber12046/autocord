import replicate
import os
from textblob import TextBlob
import dotenv

dotenv.load_dotenv(".env")

prev = ""

CHANNEL_NAME = "general"
USERNAME = "StarNumber12046"
ROLE = "admin"
SERVER_NAME = "just a server"
TASK = "talk in the server. Start by saying hi. Remember that you currently need to chat in a discord server channel (just like it was a normal chat). If there is no context present below, you will want to start by saying hi. If there is context present, follow it. Always watch your spelling."

system_prompt = """
You are currently a discord (an online chat platform) user. You can write any text and it will be sent to [channel in server]. You can also use the following commands (always prefixed by a $$):
- user: get the user id
- channel: get the channel id
- server: get the server id
- getchannels: get channels of a specified server. Output will be formatted as [channel name] - [channel type]: [channel id]. Example: test channel - text: 1234567890\ntest vc - voice: 12345678910
- getservers: get the list of the servers and IDs the user is on. Output will be formatted as [server name]: [server id]. Example: test server: 1234567890\ntest server 2: 12345678910.

To use a command, send a message in the following format: $$[command] 
Example of a command: $$server
If you send a message, do not send a command. Commands can only be used in empty messages.

Other information: \\n always is a newline, unless specified otherwise.
Any message received will be from the user
Below, under the string "START CONTEXT", the previous context is provided. It will end with the string "END PREVIOUS CONTEXT"
START CONTEXT
[prev]
END PREVIOUS CONTEXT 

Task: [task]

You are currently a user in discord and want to do what is described in the task. The task contains a description on how you want to start. Only follow it if no context is present. Otherwise, follow the previous context. Remember that you don't have to state you are a user, for the other end of the conversation not to understand you are an AI. Anything you will receive from the user(s) from now on will be a message. Messages are formatted in the following way: [USER | highest role or SYSTEM]: [message content]. All messages with the SYSTEM tag are responses to commands. Example: [SageEnjoyer | admin]: hi \n Other example: [SYSTEM]: this is a message
Whenever there is something that is dependant on the server or discord context, use commands the provided commands.
Your answers need to be formatted like this: [answer or command]. Example: hi there! How are you? \n Other example: $$server 
""".replace(
    "channel in server", f"{CHANNEL_NAME} in {SERVER_NAME}"
).replace(
    "[task]", TASK
)


def correct_sentence_spelling(sentence):
    sentence = TextBlob(sentence)
    result = sentence.correct()
    return str(result)


output = replicate.run(
    "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
    input={
        "debug": True,
        "top_k": -1,
        "top_p": 1,
        "prompt": "Start the conversation as stated by the task",
        "temperature": 0.75,
        "system_prompt": system_prompt.replace(
            "[prev]", "no context for now. This is the first message."
        ),
        "max_new_tokens": 500,
        "min_new_tokens": -1,
        "repetition_penalty": 50,
    },
)
out = "".join(output)
prev += out + "\n"
print(f"[AI]:{out}")

for x in range(5):
    # print("--------------------------")
    # print(prev)
    # print("--------------------------")
    msg = input("Enter your message.\n")
    output = replicate.run(
        "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
        input={
            "debug": True,
            "top_k": -1,
            "top_p": 1,
            "top_p": 1,
            "prompt": f"[{USERNAME} | {ROLE}]: {msg}",
            "temperature": 0.75,
            "system_prompt": system_prompt.replace("[prev]", f"{prev}"),
            "max_new_tokens": 500,
            "min_new_tokens": -1,
            "repetition_penalty": 50,
        },
    )
    out = correct_sentence_spelling("".join(output))
    print(f"[AI]:{out}")
    prev += out + "\n"
