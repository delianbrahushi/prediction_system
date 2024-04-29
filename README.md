Brahushi, Delian, 22209960

THD International Chatbot 

https://mygit.th-deg.de/db27960/speech-assistant-system


## Project description

The Chatbot for THD International Website is designed to enhance the user experience for international students at THD by providing a conversational interface to access information about upcoming events, language exchange programs, intercultural activities on campus and visa/immigration guidance.
The Chatbot has the same structure and way of conversing as showed on [[Dialog Flow](https://mygit.th-deg.de/db27960/speech-assistant-system/-/wikis/4.1-Dialogs-Flow)] 

## Installation

**Environment Details:**

- Python Version: 3.10.9
- pip Version: 23.3.1
- Flask Version: 3.0.0
- Rasa Version: 3.6.15
- Minimum Compatible Rasa Version: 3.5.0
- Rasa SDK Version: 3.6.2
- My Operating System: macOS-12.3-arm64-arm-64bit (for context)

To begin, install all required files and create a rasa project.

`rasa init`

Ensure that all files are organized correctly:

In the actions file: `actions.py` and `api_info.py`

In the data file: `nlu.yml`, `rules.yml`, and `stories.yml`

In the models directory: The latest trained model provided **OPTIONAL**

In the root directory of the chatbot: `domain.yml` and `endpoints.yml`.


Check for potential CORS issues and ensure port availability by reviewing endpoints.yml and actions.py. For example, the ports utilized in the actions file are 5055 and 5000.

Proceed by importing all necessary dependencies and modules, such as 

```from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker

from rasa_sdk.executor import CollectingDispatcher

from flask import Flask, jsonify, request

import random

import requests

```


## Basic Usage

To launch the Chatbot, follow these steps after ensuring that all files are correctly placed and prerequisites are met:

**Start Flask App:**

Open a terminal and run the Flask app on port 5000.

`python api_info.py`


In a new terminal, execute the custom actions.

`rasa run actions --actions actions`

In another terminal, train the chatbot with the following command to ensure no changes are overlooked.

`rasa train --force` 

Once the chatbot is fully trained, initiate the chatbot in the shell.

`rasa shell`

Now you can interact with the chatbot and explore its capabilities.

Possible Use Cases are provided as screenshots in the file **"Chatbot_conversation_tryout"**

## Implementation of the Requests

**actions.py:**

The `actions.py` file serves as the execution point for Python code, facilitating manipulation and interaction with the API text. It consists of two classes, each dedicated to a custom action. The first handles document retrieval, while the second implements logic to avoid repetition of possible events. This logic involves tracking the last mentioned event to prevent redundancy in subsequent inquiries. The recursion is achieved through a list that keeps tabs on the last event, and an if statement ensures it's not repeated. The file includes comments for enhanced readability.

**api_info.py:**

`api_info.py` creates a Flask app establishing a RESTFUL API with routes and paths, utilizing individual methods such as GET. The primary route fetches a random event from the dictionary and returns it as a JSON response. The additional route is designed for retrieving documents, catering to the first custom action in `actions.py`.

**nlu.yml:**

The `nlu.yml` file is crucial for defining user intents, providing a foundation for the chatbot's understanding of user inputs.

**rules.yml:**

`rules.yml` is essential for handling repetitive causes, including greetings, expressions of gratitude, and farewells, enhancing the chatbot's conversational flow.

**stories.yml:**

The `stories.yml` file is vital for orchestrating the dialog flow, ensuring a cohesive and effective conversation route.

**domain.yml:**

`domain.yml` is necessary for defining chatbot utterances and potential responses, contributing to a diverse and engaging interaction.

**Unused Files:**

- `config.yml` and `credentials.yml` were not employed in this project.

**endpoints.yml:**

The details regarding `endpoints.yml` are thoroughly explained in the "Basic Usage" section.



## Work done

Delian Brahushi has implemented all

## Copyright Claim

All components and content within this project have been implemented in accordance with ethical standards and best practices. The project has been developed with a commitment to integrity and compliance with relevant guidelines.

The assistance of ChatGPT has been utilized exclusively for rewriting, formulating, and ensuring grammatical correctness in the textual content. The usage of ChatGPT aims to enhance the clarity and coherence of the project's documentation and user interactions. 

**ChatGPT has not been used to perform any substantive work other than enhancing the textual context**, and all creative and functional aspects of the project are independently developed by me.

The images used in the personas of the chatbot are sourced from copyright-free repositories and obtained from [Image Website](https://unsplash.com).

I take full disclosure of my work, emphasizing transparency and adherence to intellectual property rights. 




