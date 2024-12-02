from model import generate
import PyPDF2

file_path = input("enter file path : ")
with open(file_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    skills_projects_from_resume = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        skills_projects_from_resume += page.extract_text()


prompt_1 = """
Extract the skills and projects from the following text. Organize the skills under a "Skills" section and
projects under a "Projects" section, providing clear bullet points for each.

{Skills_and_projects}
"""

prompt1 = prompt_1.format(Skills_and_projects=skills_projects_from_resume)

prompt_2 = """
Given the following skills and projects, generate a  one simple interview question with answer based on the skills mentioned below. Make the answers conversational and relatable, as if spoken by someone during a casual interview.
{skills}
###
Example:
1. What programming languages do you know, and how do you decide which one to use for a project?
Answer: I have experience with JavaScript, Python, and Java. When it comes to choosing a language, I think about the project requirements and what the team feels comfortable with. For instance, if we're working on a web application, I usually lean toward JavaScript or Python because they fit well with the task at hand. It’s all about picking the right tool for the job!
###
"""
extracted_skills_projects = generate(prompt1, model, tokenizer)  # Extracting  skills and projects details
prompt2 = prompt_2.format(skills=extracted_skills_projects)
question_text = generate(prompt2, model, tokenizer)  # Generating Question


prompt_3 = """
You are an evaluator tasked with comparing the user answer with given question: Your goal is to assess the quality of the answer based on clarity, relevance, and completeness.

1. Read user answer carefully.
2. Compare the user's answer and the question based on the following criteria:
   - Clarity: Is the answer easy to understand?
   - Relevance: Does the answer address the question appropriately?
   - Completeness: Does the answer provide enough detail and information?

3. Provide a score from 0 to 10 for each criterion for user answer , where:
   - 0 means poor quality
   - 5 means average quality
   - 10 means excellent quality

4. After scoring, provide an overall score (out of 10) based on the combined evaluation of clarity, relevance, and completeness.

### Question:
{Question}

### User's Answer:
"{user_answer}"

### Evaluation:
- Clarity Score:
- Relevance Score:
- Completeness Score:
- Overall Score:
"""

question_part = question_text[: question_text.index("Answer")]
question_section = question_part.strip()
print("Generated Question")
print(question_section.replace("*", ""))
user_answer = input("Enter Your Answer :")
prompt3 = prompt_3.format(Question=question_section, user_answer=user_answer)
# Evaluation  of User Answer
print(generate(prompt3, model, tokenizer).replace("*", ""))
