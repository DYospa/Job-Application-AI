from ai.candidate_profile import build_candidate_profile, CandidateProfile

SAMPLE_RESUME_DICT = {
    "name": "Doron Yospa",
    "email": "doronyospa@gmail.com",
    "phone": "(555) 123-4567",
    "skills": ["Python", "SQL", "python", "  ", "Machine Learning"],  # includes dupe/blank on purpose
    "education": {"school": "Georgia State University", "degree": "B.S. Data Science", "graduation": "2026"},
    "experience": [{"title": "Claims Specialist", "company": "State Farm", "description": []}],
    "projects": [{"title": "Job Scheduling Assistant"}],
    "certifications": ["Google Professional Data Analytics Certificate"],
}


def test_build_candidate_profile_basic_fields():
    profile = build_candidate_profile(SAMPLE_RESUME_DICT)
    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Doron Yospa"
    assert profile.email == "doronyospa@gmail.com"
    assert profile.education["school"] == "Georgia State University"


def test_build_candidate_profile_dedupes_skills_case_insensitively():
    profile = build_candidate_profile(SAMPLE_RESUME_DICT)
    # "Python" and "python" should collapse to one entry; blank entries dropped.
    assert profile.skills.count("Python") + profile.skills.count("python") == 1
    assert "" not in profile.skills


def test_build_candidate_profile_handles_empty_input():
    profile = build_candidate_profile({})
    assert profile.name == ""
    assert profile.skills == []
    assert profile.education == {}


def test_to_dict_roundtrip():
    profile = build_candidate_profile(SAMPLE_RESUME_DICT)
    d = profile.to_dict()
    assert d["name"] == "Doron Yospa"
    assert isinstance(d["skills"], list)
