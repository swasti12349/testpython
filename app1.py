import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import PyPDF2
import docx
import os
import openai
from dotenv import load_dotenv
import re
import openai
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from pathlib import Path
import plotly.graph_objects as go

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
openai_model_name = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')  # Default to gpt-3.5-turbo
print("asdasd", openai_model_name)
 

# OpenAI API key
openai.api_key = openai_api_key
print("openai key", openai_api_key)

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Helper function to extract text from Word document
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text



def extract_skills_from_jd(job_description):

    prompt = (
        f"Extract only skills required from this JD and dont add eny extra word apart from the JD text. Just give me a list of technology without any special characters."
        f"{job_description}"
    )

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        
        # Parse the response content
        response_text = response['choices'][0]['message']['content']
        # print("text...  " + response_text)
        # Extract potential skills using regex and cleanup
        skills = re.split(r',|\n|-', response_text)  # Split by commas, newlines, or hyphens
        skills = [skill.strip() for skill in skills if skill.strip()]  # Remove extra whitespace
        return skills  # Limit to max_skills

    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []


def extract_skills_from_resume(job_description, max_skills=20):

    prompt = (
        f"Extract only skills keywords from the resume"
        f"{job_description}"
    )

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        
        # Parse the response content
        response_text = response['choices'][0]['message']['content']
        # print("text in resume...  " + response_text)
        # Extract potential skills using regex and cleanup
        skills = re.split(r',|\n|-', response_text)  # Split by commas, newlines, or hyphens
        skills = [skill.strip() for skill in skills if skill.strip()]  # Remove extra whitespace
        return skills[:max_skills]  # Limit to max_skills

    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []




def extract_experience_from_jd(job_description):
    
    prompt = f"""    
    Job Description:
    {job_description}
    
    Only give how much experience required in number.
    """
    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "system", "content": "You are an expert job description analyzer."},
                      {"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        keywords = response["choices"][0]["message"]["content"].strip()
        return set(map(str.strip, keywords.split(',')))
    except Exception as e:
        print(f"Error in extracting experience keywords: {e}")
        return set()
    
    
def extract_experience_from_resume(job_description):
    
    prompt = f"""    
    Job Description:
    {job_description}
    
    Only give how much experience does this candidate has in number.
    """
    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        keywords = response["choices"][0]["message"]["content"].strip()
        return set(map(str.strip, keywords.split(',')))
    except Exception as e:
        print(f"Error in extracting experience keywords: {e}")
        return set()

def extract_education_from_resume(job_description):
    
    prompt = f"""    
    Job Description:
    {job_description}
    
    Only give how much education does this candidate has.
    """
    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        keywords = response["choices"][0]["message"]["content"].strip()
        return set(map(str.strip, keywords.split(',')))
    except Exception as e:
        print(f"Error in extracting experience keywords: {e}")
        return set()
    
    
def extract_education_from_jd(job_description):
    
    prompt = f"""
    What educations required for this job just give me the degrees.
    
    Job Description:
    {job_description}
    
    """
    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        keywords = response["choices"][0]["message"]["content"].strip()
        return set(map(str.strip, keywords.split(',')))
    except Exception as e:
        print(f"Error in extracting education keywords: {e}")
        return set()


def extract_soft_skills_from_jd(job_description):
    """
    Extract soft skills-related keywords from the Job Description using OpenAI.
    """
    prompt = f"""
    Extract all soft skills-related keywords or phrases from the following Job Description. Focus on terms like "communication", "teamwork", "leadership", "problem-solving", "creativity", or similar:
    
    Job Description:
    {job_description}
    
    Provide the keywords or phrases as a comma-separated list.
    """
    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[{"role": "system", "content": "You are an expert job description analyzer."},
                      {"role": "user", "content": prompt}],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        keywords = response["choices"][0]["message"]["content"].strip()
        return set(map(str.strip, keywords.split(',')))
    except Exception as e:
        print(f"Error in extracting soft skills: {e}")
        return set()


def calculate_skill_match_percentage(skill_keywords, resume_keywords):
    """
    Calculate the skill match percentage between job description (JD) skills and resume skills.

    :param skill_keywords: Set of required skills in the job description (JD).
    :param resume_keywords: Set of skills extracted from the resume.
    :return: Skill match percentage.
    """
    # Normalize resume keywords by filtering only those present in skill_keywords
    matched_skills = skill_keywords.intersection(resume_keywords)
    print("mathced skill", matched_skills)
    # Calculate percentage
    total_skills_in_jd = len(skill_keywords)
    matching_skills_count = len(matched_skills)

    if total_skills_in_jd == 0:
        return 0.0  # Avoid division by zero

    match_percentage = (matching_skills_count / total_skills_in_jd) * 100

    return round(match_percentage, 2), matched_skills  # Returning rounded percentage

def calculate_exp_match_score(experience_keywords, experience_from_resume):
    
    # Convert sets to comma-separated lists
    exp_from_jd = ", ".join(experience_keywords)
    resume_exp_str = ", ".join(experience_from_resume)

    prompt = f"""
    

    - **Experience from JD**: {exp_from_jd}
    - **Resume experience**: {resume_exp_str}
    Calculate how much the experience is fulfilled if the 
    resume experience is more than the JD give 100% else calculate and give the percentage.
    Method:
    Convert the years into months and calculate the percentage.
    """

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[
                
                {"role": "user", "content": prompt}
            ],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        
        # Extract and convert the response to float
        match_percentage = response["choices"][0]["message"]["content"].strip()
        return match_percentage

    except Exception as e:
        print(f"Error calculating skill match percentage: {e}")
        return 0.0  # Default to 0% in case of an error


def calculate_edu_match_score(edu_keywords, edu_from_resume):
    
    # Convert sets to comma-separated lists
    edu_from_jd = ", ".join(edu_keywords)
    resume_edu_str = ", ".join(edu_from_resume)

    prompt = f"""
    

    - **Education from JD**: {edu_from_jd}
    - **Resume Education**: {resume_edu_str}
    Calculate how much the education is fulfilled if the 
    JD education is thier in the resume education then give 100.0% else give 0%
    and dont give score for the cgpa. Just give me the percentage. either 0% or 100%.
    """

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[
                
                {"role": "user", "content": prompt}
            ],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        
        # Extract and convert the response to float
        match_percentage = response["choices"][0]["message"]["content"].strip()
        return match_percentage

    except Exception as e:
        print(f"Error calculating skill match percentage: {e}")
        return 0.0  # Default to 0% in case of an error
        
        
def calculate_soft_skill_match_score(softskilljd, resumesoftskill):
    
    # Convert sets to comma-separated lists
    var_softskilljd = ", ".join(softskilljd)
    var_resumesoftskill = ", ".join(resumesoftskill)

    prompt = f"""
    

    - **Soft Skill from JD**: {var_softskilljd}
    - **Resume Soft Skill**: {var_resumesoftskill}
    Calculate how much the soft skill is matching just give me the percentage.
    """

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[
                
                {"role": "user", "content": prompt}
            ],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        
        # Extract and convert the response to float
        match_percentage = response["choices"][0]["message"]["content"].strip()
        return match_percentage

    except Exception as e:
        print(f"Error calculating soft skill match percentage: {e}")
        return 0.0  # Default to 0% in case of an error

    
def extract_percentage(text):
    match = re.search(r"(\d+\.\d+)%", text)
    return float(match.group(1)) if match else 0.0

def extract_percent_ai_model(text):
    

    prompt = f"""
    
    extract the final percentage from the text and return the final percentage only.
    And just say "The final percent is xx.yy%"
    The text: {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model=openai_model_name,
            messages=[
                
                {"role": "user", "content": prompt}
            ],
            temperature = 0,  # Low randomness for deterministic results
            top_p = 1,  # Moderate nucleus sampling for controlled diversity
            frequency_penalty = 0,  # Slight penalty to avoid repetitive answers
            presence_penalty = 0.1,  # Ensures all skills are considered
            max_tokens = 500,  # Restrict output length for predictability
        )
        
        # Extract and convert the response to float
        match_percentage = response["choices"][0]["message"]["content"].strip()
        return match_percentage

    except Exception as e:
        print(f"Error finding percentage: {e}")
        return 0.0  # Default to 0% in case of an error
 

def calculate_ats_score(resume_text, job_description, criteria_weights):
    if not resume_text:
        return 0.0, "No common skills has matched", "The resume is blank or has unrelated data"
    skill_keywords = set(extract_skills_from_jd(job_description))
    print("skill from jd", skill_keywords)
    resume_keywords = set(extract_skills_from_resume(resume_text))
    print("skill from resumee", resume_keywords)

    # Normalize resume skills (clean extra text formatting)
    normalized_resume_keywords = {skill.split(":")[-1].strip() for skill in resume_keywords}

    # Calculate match percentage
    skill_match_percentage, matchedSkills = calculate_skill_match_percentage(skill_keywords, normalized_resume_keywords)
    
    # matched_skill = calculate_skill_match_percentage(skill_keywords, normalized_resume_keywords)
    print(f"Skill Match Percentage: {skill_match_percentage}%")
    skill_final_score_prev = extract_percent_ai_model(skill_match_percentage)
    skill_final_score = extract_percentage(skill_final_score_prev)
    
    print("Skill.. ", skill_final_score)
    experience_keywords = extract_experience_from_jd(job_description)
    experience_from_resume = extract_experience_from_resume(resume_text)
    experience_match_score = calculate_exp_match_score(experience_keywords, experience_from_resume)
    exp_final_score_prev = extract_percent_ai_model(experience_match_score)
    print("expre final scor eprev..", exp_final_score_prev)
    exp_final_score = extract_percentage(exp_final_score_prev)
    print("exp...", exp_final_score)
    # print("experience..", exp_final_score_prev)
    education_keywords = extract_education_from_jd(job_description)
    education_from_resume = extract_education_from_resume(resume_text)
    edu_score = calculate_edu_match_score(education_keywords, education_from_resume)
    edu_final_score_prev = extract_percent_ai_model(edu_score)

    edu_final_score = extract_percentage(edu_final_score_prev)
    print("edu...", edu_final_score)
    soft_skills = extract_soft_skills_from_jd(job_description)
    soft_skill_score = calculate_soft_skill_match_score(soft_skills, resume_keywords)
    soft_skill_final_score = extract_percentage(soft_skill_score)
    # print("softskillssssss...", soft_skill_final_score)
    # soft_skills_match = soft_skills & resume_keywords
    # soft_skill_score = soft_skills_match
    # soft_skill_score = (len(soft_skills_match) / len(soft_skills)) * criteria_weights.get("Soft Skills *", 0) if soft_skills else 0
    print("soft...", soft_skill_score)
    # common_skills = skill_keywords & resume_keywords
    # Sum the weighted scores


    skill_weight = criteria_weights.get("Skills *", 0)
    exp_weight = criteria_weights.get("Experience *", 0)
    edu_weight = criteria_weights.get("Education *", 0)
    soft_skill_weight = criteria_weights.get("Soft Skills *", 0)

    skill_percentage_final = skill_final_score * (skill_weight / 100)
    exp_percentage_final = exp_final_score * (exp_weight / 100)
    edu_percentage_final = edu_final_score * (edu_weight / 100)
    soft_skill_percentage_final = soft_skill_final_score * (soft_skill_weight / 100)

    total_score = skill_percentage_final + exp_percentage_final + edu_percentage_final + soft_skill_percentage_final

    final_score = round(total_score, 2)
    return round(final_score, 2), matchedSkills, skill_keywords, round(skill_percentage_final, 2), round(exp_percentage_final, 2), round(edu_percentage_final, 2), round(soft_skill_percentage_final, 2)

# Add this function to check API key validity
def validate_api_key(api_key):
    try:
        # Test the API key by making a simple request
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}]
        )
        return True
    except openai.error.AuthenticationError:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def extract_text_from_txt(file_path):
    """
    Reads and returns the content of a .txt file.

    :param file_path: Path to the .txt file
    :return: Text content as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

                    
def generate_resume_insights(
    resume_text,
    selected_criteria,
    criteria_weights,
    skill_percentage_final,
    exp_percentage_final,
    edu_percentage_final,
    soft_skill_percentage_final
):
    # Build criteria description safely
    criteria_description = ', '.join(
        [f"{criterion} ({criteria_weights.get(criterion, 0)}%)" for criterion in selected_criteria]
    )
    
    # Use .get() for each weight inside the prompt to avoid KeyError
    prompt = f"""Analyze the following resume content based on the selected criteria and their weightage: {criteria_description}.
    Resume content:
    {resume_text}

    And don't give the final score. Also calculate the soft skill percentage. These are the final scores obtained for the given weightage; show them as well in the headline.

    1. Skills: {skill_percentage_final} / {criteria_weights.get("Skills *", 0)}
    2. Experience: {exp_percentage_final} / {criteria_weights.get("Experience *", 0)}
    3. Education: {edu_percentage_final} / {criteria_weights.get("Education *", 0)}
    4. Soft Skills: {soft_skill_percentage_final} / {criteria_weights.get("Soft Skills *", 0)}

    This is the response format:

    Skills: {skill_percentage_final}/{criteria_weights.get("Skills   *", 0)} (XX%)    
        Strengths: The candidate demonstrates a strong skill set in programming languages (Java, SQL, JavaScript, HTML, C++), modern technologies (AWS, Git/Github, Figma), and frameworks (ReactJs, NextJs, Springboot, NodeJs). The inclusion of both frontend and backend technologies indicates a well-rounded skill set suitable for software development roles.
        Areas for Improvement: None noted; the skills section is comprehensive and relevant to the industry.

    Experience: {exp_percentage_final}/{criteria_weights.get("Experience *", 0)} (XX%)
        Strengths: The candidate has completed a 2-month internship as a Web Developer, which provides some practical experience in web development and collaboration.
        Areas for Improvement: The experience is limited to a short internship, and there are no additional work experiences or internships listed. Gaining more hands-on experience or contributing to open-source projects could enhance this section.

    Education: {edu_percentage_final}/{criteria_weights.get("Education *", 0)} (XX%)
        Strengths: The candidate is currently pursuing a B.Tech in Computer Science Engineering with a commendable CGPA of 8.44. The educational background includes a solid foundation from high school, with good scores in both matriculation and higher secondary education.
        Areas for Improvement: While the education section is strong, the candidate could benefit from mentioning any relevant coursework, honors, or extracurricular activities related to software development.

    Soft Skills: {soft_skill_percentage_final}/{criteria_weights.get("Soft Skills *", 0)} (XX%)
        Strengths: The candidate mentions strong collaboration and problem-solving skills developed during the internship.
        Areas for Improvement: There is a lack of explicit mention of other soft skills such as communication, teamwork, adaptability, or leadership. Including specific examples or experiences that demonstrate these skills would strengthen this section.

    Summary:

        Skills: XX.XX
        Experience: XX.XX
        Education: XX.XX
        Soft Skills: XX.XX

    Overall, the candidate has a strong foundation in skills and education but needs to enhance their experience and soft skills sections to present a more balanced profile for potential employers.
    """

    response = openai.ChatCompletion.create(
        model=openai_model_name,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.1,
        max_tokens=500,
        messages=[
            {"role": "system", "content": "You are an expert job matching assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


# Streamlit UI
st.set_page_config(page_title="ATS Resume Screening App", page_icon="nathcorp.jpg", layout="centered")
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 5 MB in bytes

st.markdown(
    """
    <div style="background-color: white; padding: 1rem; text-align: center;">
        <img src="https://nathcorp.com/wp-content/uploads/2020/01/NathcorpLogo-Text-side_400x53.png" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("ATS Resume Screening App (Preview)", anchor=False)

st.sidebar.title("Select Criteria and Enter Weightage")

# Hide the left arrow (sidebar collapse/expand control)
hide_sidebar_arrow = """
    <style>
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_sidebar_arrow, unsafe_allow_html=True)

criteria_options = ["Skills *", "Experience *", "Education *", "Soft Skills *"]
criteria_value = [30, 15, 40, 15]

# Tooltip messages for each criterion
criteria_tooltips = {
    "Skills *": "Adjust skills to increase or decrease the importance of skills in the final ATS score.",
    "Experience *": "Adjust Experience to increase or decrease the importance of Experience in the final ATS score.",
    "Education *": "Adjust Education to increase or decrease the importance of Education in the final ATS score.",
    "Soft Skills *": "Adjust Soft Skills to increase or decrease the importance of Soft Skills in the final ATS score."
}

selected_criteria = []
criteria_weights = {}

for i, criterion in enumerate(criteria_options):
    selected = st.sidebar.checkbox(criterion, True)
    if selected:
        weight = st.sidebar.number_input(
            f"Weightage for {criterion} (%)",
            value=int(criteria_value[i]),
            min_value=0,
            step=1,
            format="%d",
            help=criteria_tooltips.get(criterion, "")  # Add tooltip here
        )
        criteria_weights[criterion] = weight
        selected_criteria.append(criterion)

st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #26273;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ccc;
    }
    </style>
    <div class="footer">
        © 2025 Nathcorp | All rights reserved | v1.2

    </div>
    """, unsafe_allow_html=True)



total_weight = sum(criteria_weights.values())
if total_weight > 100:
    st.sidebar.warning("The total weightage exceeds 100%! Please adjust the values.")
elif total_weight < 100:
    st.sidebar.info("The total weightage is less than 100%. You can allocate more.")



job_description = st.text_area("Enter the Job Description (JD): *", height=200, help="Enter the job description to analyze resumes against.")
uploaded_resumes = st.file_uploader(label="", type=['pdf', 'docx', '.doc', '.txt'], accept_multiple_files=True)
# uploaded_resumes = ""

# with open("ind.html", "r") as f:
#     html_uploader = f.read()

# components.html(html_uploader, height=100)

if not validate_api_key(openai_api_key):
    st.error("The OpenAI API key is invalid or expired. Please update your API key.")
else:
    if uploaded_resumes and job_description:
 

        if total_weight == 100 and job_description:
            if st.button("Submit"):
                st.subheader("Resume Analysis", anchor=False)

                summary_data = []  # For storing table data

                for uploaded_resume in uploaded_resumes:
                    uploaded_resume.seek(0, 2)
                    file_size = uploaded_resume.tell()
                    uploaded_resume.seek(0)

                    if file_size > MAX_FILE_SIZE_BYTES:
                        st.warning(f"File {uploaded_resume.name} is larger than {MAX_FILE_SIZE_MB} MB, please upload a file under 5 MB.")
                        st.stop()

                    # Extract text based on file type
                    if uploaded_resume.type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_resume)
                    elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordmlprocessing.document":
                        resume_text = extract_text_from_docx(uploaded_resume)
                    elif uploaded_resume.type == "text/plain":
                        resume_text = extract_text_from_txt(uploaded_resume)
                    else:
                        resume_text = None

                    if resume_text:
                        with st.spinner(f"Analyzing {uploaded_resume.name}..."):
                            skills_words_from_jd = extract_skills_from_jd(job_description)

                            final_score, common_skills, skill_keywords, skill_percentage_final, exp_percentage_final, edu_percentage_final, soft_skill_percentage_final = calculate_ats_score(
                                resume_text, job_description, criteria_weights
                            )
                            insights = generate_resume_insights(
                                resume_text, selected_criteria, criteria_weights, skill_percentage_final, exp_percentage_final, edu_percentage_final, soft_skill_percentage_final
                            )

                            # Append to summary table
                            summary_data.append({
                                "Resume Name": uploaded_resume.name,
                                "Skill %": f"{skill_percentage_final:.2f}",
                                "Experience %": f"{exp_percentage_final:.2f}",
                                "Education %": f"{edu_percentage_final:.2f}",
                                "Soft Skill %": f"{soft_skill_percentage_final:.2f}",
                                "Final ATS Score": f"{final_score:.2f}"
                            })

                            # Success card
                            st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#d4edda; border:1px solid #c3e6cb; color:#155724; margin-bottom:15px;">Analysis Completed for <strong>{uploaded_resume.name}</strong>!</div>""", unsafe_allow_html=True)

                            # Skills Card
                            st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#cce5ff; border:1px solid #b8daff; color:#004085; margin-bottom:15px;">
                                <h3>Skills</h3>
                                <strong>Matched Skill Keywords:</strong> {', '.join(common_skills)}<br>
                                <strong>Extracted Skill Keywords from JD:</strong> {', '.join(skills_words_from_jd)}<br>
                                <strong>Skill Match Percentage:</strong> {skill_percentage_final:.2f}%
                            </div>""", unsafe_allow_html=True)

                            # Experience Card
                            st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#cce5ff; border:1px solid #ffeeba; color:#004085; margin-bottom:15px;">
                                <h3>Experience</h3>
                                <strong>Experience Match Percentage:</strong> {exp_percentage_final:.2f}%
                            </div>""", unsafe_allow_html=True)

                            # Education Card
                            st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#cce5ff; border:1px solid #d6d8db; color:#004085; margin-bottom:15px;">
                                <h3>Education</h3>
                                <strong>Education Match Percentage:</strong> {edu_percentage_final:.2f}%
                            </div>""", unsafe_allow_html=True)

                            # Soft Skills Card
                            st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#cce5ff; border:1px solid #f5c6cb; color:#004085; margin-bottom:15px;">
                                <h3>Soft Skills</h3>
                                <strong>Soft Skills Match Percentage:</strong> {soft_skill_percentage_final:.2f}%
                            </div>""", unsafe_allow_html=True)

                            # GPT Insights Card
                            st.markdown(f"""<div style="padding:15px; border-radius:10px; background-color:#cce5ff; border:1px solid #fff3cd; color:#004085; margin-bottom:15px;">
                                <h3>GPT Insights</h3>
                                {insights}
                            </div>""", unsafe_allow_html=True, )

                            # Final ATS Score Card  
                            st.markdown(f"""<div style="padding:20px; border-radius:12px; background-color:#000080; color:white; text-align:center; font-size:1.8em; margin-bottom:10px;">
                                Final ATS Score for {uploaded_resume.name}: {final_score:.0f}%
                            </div>""", unsafe_allow_html=True)
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=final_score,
                                title={'text': f"ATS Score - {uploaded_resume.name}"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 75], 'color': "gray"},
                                        {'range': [75, 100], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)

                            

                # Display summary table after all resumes are processed
                if summary_data:
                    st.markdown("### 📊 Resume Summary Table",)
                    st.dataframe(summary_data)
                                        # Generate PDF report from summary_data
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(200, 10, "Resume Summary Report", ln=True, align="C")

                    pdf.set_font("Arial", "", 12)
                    pdf.ln(10)
                    headers = list(summary_data[0].keys())
                    for row in summary_data:
                        for key in headers:
                            pdf.cell(0, 10, f"{key}: {row[key]}", ln=True)
                        pdf.ln(5)

                    pdf_output_path = "resume_summary.pdf"
                    pdf.output(pdf_output_path)

                    # Add a download button
                    if Path(pdf_output_path).exists():
                        with open(pdf_output_path, "rb") as f:
                            st.download_button(
                                label="📄 Download Resume Summary PDF",
                                data=f,
                                file_name="resume_summary.pdf",
                                mime="application/pdf"
                            )

# Add a blue Logout button in the top left
logout_css = '''
    <style>
    .logout-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 9999;
    }
    .logout-btn button {
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-size: 1.1em;
        border: none;
        cursor: pointer;
        transition: background 0.2s;
    }
    .logout-btn button:hover {
        background-color: #155a8a;
    }
    </style>
'''
st.markdown(logout_css, unsafe_allow_html=True)

logout_clicked = st.markdown(
    '''<div class="logout-btn"><form action="#" method="post"><button type="submit" name="logout">Logout</button></form></div>''',
    unsafe_allow_html=True
)

# Streamlit workaround for HTML button: use query params or session state
if st.session_state.get('logout', False):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Add a real Streamlit button for logout logic (hidden, but triggers on HTML click)
if st.sidebar.button("_hidden_logout", key="_hidden_logout", help="hidden", args=()):
    st.session_state['logout'] = True
    st.rerun()

