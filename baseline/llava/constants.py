#!/usr/bin/env python3

ADDITIONAL_PUZZLES = [1, 8, 9, 33, 57, 63, 64, 70, 81, 90]

#  (18    order), (61  algebra), (62    logic), (69  spatial),
#  (71     math), (73     path), (77  pattern), (94  measure),
#  (99 counting)
INSTANCE_SPLIT_PUZZLES = [18, 61, 62, 69, 71, 73, 77, 94, 99]

PUZZLE_SPLIT_PUZZLES = [61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 94, 95, 96, 97, 98, 99, 101]

MAX_PUZZLE_ID = 101

LLAVA_PROMPT_TEMPLATE = "[INST] <image>\n{} [/INST]"

LLAVA_ANSWER_TEMPATE = "Please provide the answer in a single upper case letter, e.g: A"

SAMPLE_QUESTION = """
    Five polygon cards are stacked on a table as shown. Each card is numbered according to the color shown at the bottom.
    If we remove the cards one by one from the top of the stack, in what possible order are the cards removed?
    A: 4-5-3-1-2,B: 2-4-5-3-1,C: 2-4-1-3-5,D: 1-4-3-2-5,E: 1-5-4-2-3
    """

SAMPLE_OUTPUT = """
{
"sub_questions": [
    {
    "question": "How many cards are in the stack?",
    "answer": "[Answer based on the given image]"
    },
    {
    "question": "What is the number on the top card of the stack?",
    "answer": "[Answer based on the given image]"
    },
    {
    "question": "What is the number on the card immediately below the top card?",
    "answer": "[Answer based on the given image]"
    },
    {
    "question": "What is the number on the card in the middle of the stack?",
    "answer": "[Answer based on the given image]"
    },
    {
    "question": "What is the number on the second to last card in the stack?",
    "answer": "[Answer based on the given image]"
    },
    {
    "question": "What is the number on the bottom card of the stack?",
    "answer": "[Answer based on the given image]"
    }
]
}
"""

SAMPLE_CAPTION_OUTPUT = """
{
  "caption": "The image depicts five overlapping square cards, each of a different color and labeled with numbers from 1 to 5. The color of each card corresponds to the numbers provided in the key below the stack: 1. Light purple (Card 1), 2. Light blue (Card 2), 3. Dark green (Card 3), 4. Purple (Card 4), 5. Gold (Card 5). The cards are stacked in such a way that some cards are partially covered by others. Specifically, the order of the cards from top to bottom, based on the overlapping areas, appears to be: 1. The purple card (4) is on top, 2. Underneath the purple card is the gold card (5), 3. Below the gold card is the dark green card (3), 4. The light purple card (1) is next, 5. The light blue card (2) is at the bottom."
}
"""

GENERATE_SUB_QUESTION_PROMTP = """
  Given the complex question '{}', break it down into critical, simple and necessary sub-questions and their corresponding answers.
  Each sub-question only requests single information without reasoning.
  With the sub-questions and its answer, we can easily resolve the complex question.

  For example, if asked about '{}': you might generate sub-questions and its answers in JSON format as follows:
  {}.
  Output the answer in pure JSON format, without the format prefix like ```json.
  """

GENERATE_CAPTION_PROMTP = """
  Please generate a detailed caption for the image including objects, scenes, actions, and any other relevant details to allow others
  depends on the generated caption to answer the following question, but you should not directly include the answer in the caption. 
  If the input image includes nothing (a pure white image), just says: The image is a blank image, answer the question based on the question itself.
  '{}'

  For example, if asked about '{}': you might generate caption in JSON format as follows:
  {}

  Output the answer in pure JSON format, without the format prefix like ```json.
  """

BASELINE_SUB_QUESTION_PROMTP = """
  Given the following previous answered questions and input image, there could be no answered question when it's first time to ask:
  {}

  Please answer the following new question and give the answer as short as possible. If the answer is a number, provide a numberical answer e.g. 1, 22, 3.

  {}
  """

BASELINE_CAPTION_PROMTP = """
  Given the following detailed caption for the input image:
  {}

  Please answer the following question:
  {}
  """
  
  
# TOT Prompt
TOT_SIMPLE_QUESTION = """
There are 29 people standing in line, including Mortie and Goober. Mortie knows that there are 2 people in front of him. If Mortie is in front of Goober and there are 16 people between Mortie and Goober, how many of the people in the line are behind Goober? A: 10,B: 9,C: 12,D: 8,E: 7.
"""

TOT_SIMPLE_OUTPUT = """
{
  "sub_questions": [
      {
      "question": "How many people are in front of Mortie?",
      "answer": "2"
      },
      {
      "question": "How many people are between Mortie and Goober?",
      "answer": "16"
      },
      {
      "question": "How many people are standing in line?",
      "answer": "29"
      }
  ]
}
"""

TOT_GENERATE_SUB_QUESTION_PROMTP = """
Given the complex question '{}' and its corresponding image(attached), 
please generate exactly three simple sub questions and their corresponding answers, consider different starting points to resolve the complex problem. 
Each sub-question only requests single information without reasoning.

For example, if asked about '{}': you might generate sub-questions and its answers in JSON format as follows:
{}.
Output the answer in pure JSON format, without the format prefix like ```json.
"""

TOT_FOLLOW_UP_OUTPUT = """
{
  "follow_up_questions": [
      {
        "question": "How many people are between Mortie and Goober?",
        "answer": "16"
      }
  ]
}
"""

TOT_FINAL_OUTPUT = """
{
  "follow_up_questions": []
  "final_answer": "A"
}
"""

TOT_GENERATE_FOLLOW_UP_QUESTION_PROMTP = """
Given the previous steps of questions and answers:
{}

Please generate the next follow-up step (a single follow-up question and its answer) to help move closer to answer the following question:
{}

Ensure that each follow-up question is logically connected to the previous answer and provides progress toward resolving the complex question.

For example, if asked about: 

Given the previous steps of questions and answers:
Step 0, Question:How many people are in front of Mortie?  Answer:2

To answer the original complex question:
{}

You might generate the follow-up question and its answer in JSON format as follows:
{}

Please stop generating new steps if you think there is enough information to answer the original question or the total depth of the follow-up questions is over than {},
return an empty "follow_up_questions" list and a single letter option answer for the original question in "final_answer" in JSON format as follows:
{}

Output the answer in pure JSON format, without the format prefix like ```json.
"""

SIMPLE_COT_OUTPUT = """
{
  "reasoning_steps": "Detailed explanation of how the correct answer was determined.",
  "final_answer": "A"
}
"""

SIMPLE_COT_PROMPT = """<image>
Let's answer the following multiple-choice questions step by step.
{}

Please provide detailed reasoning and choose a final answer from the given options. 
Format your response in pure JSON as follows without the format prefix like ```json.
In "reasoning_steps", include a description of the logical reasoning that leads to the answer.
In "final_answer", provide the selected option as a single letter. 

For example:
{}

"""