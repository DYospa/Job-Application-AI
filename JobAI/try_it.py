from ai.resume_parser import parse_resume 
from ai.candidate_profile import build_candidate_profile 
import json 
resume = parse_resume("DoronResume.pdf") 
profile = build_candidate_profile(resume) 
print(json.dumps(profile.to_dict(), indent=2))
