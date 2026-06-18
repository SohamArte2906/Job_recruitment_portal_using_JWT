from fastapi import FastAPI, Request, Form, Depends, Response, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.routing import APIRouter
from sqlalchemy import create_engine, String, select, Float, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session
from pydantic import BaseModel
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
import os
import shutil
from typing import Optional

# ==========================================
# 1. SECURITY & JWT CONFIGURATION
# ==========================================
SECRET_KEY = "my_super_secret_key_for_development"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8')[:72], hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8')[:72], bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ==========================================
# 2. DATABASE SETUP
# ==========================================
engine = create_engine("sqlite:///recruitment.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), default="candidate")

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(150))
    company: Mapped[str] = mapped_column(String(100))
    location: Mapped[str] = mapped_column(String(100))
    job_type: Mapped[str] = mapped_column(String(50))
    salary_range: Mapped[str] = mapped_column(String(50))
    description: Mapped[str] = mapped_column(String(2000))
    required_skills: Mapped[str] = mapped_column(String(500))
    experience_level: Mapped[str] = mapped_column(String(50))
    posted_by: Mapped[int] = mapped_column(Integer)
    posted_at: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20), default="open")

class Application(Base):
    __tablename__ = "applications"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(Integer)
    candidate_id: Mapped[int] = mapped_column(Integer)
    candidate_name: Mapped[str] = mapped_column(String(100))
    candidate_email: Mapped[str] = mapped_column(String(100))
    resume_path: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    cover_letter: Mapped[str] = mapped_column(String(2000))
    skills: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(30), default="Applied")
    match_score: Mapped[str] = mapped_column(String(10), default="0")
    applied_at: Mapped[str] = mapped_column(String(30))
    interview_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    interview_notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

Base.metadata.create_all(bind=engine)

# ==========================================
# 3. FASTAPI SETUP & SHARED DEPENDENCIES
# ==========================================
app = FastAPI(title="TalentBridge API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="Frontend")

bearer_scheme = HTTPBearer(auto_error=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _decode_token(token: str) -> Optional[str]:
    """Decode JWT and return email, or None on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.InvalidTokenError:
        return None

# ── HTML dependency: reads from cookie ──────────────────────────────────────
def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    email = _decode_token(token)
    if not email:
        return None
    return db.scalars(select(User).where(User.email == email)).first()

# ── API dependency: reads from Authorization: Bearer <token> ─────────────────
def get_api_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    email = _decode_token(credentials.credentials)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.scalars(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def compute_match_score(job_skills: str, candidate_skills: str) -> int:
    job_set = set(s.strip().lower() for s in job_skills.split(",") if s.strip())
    cand_set = set(s.strip().lower() for s in candidate_skills.split(",") if s.strip())
    if not job_set:
        return 0
    return round((len(job_set & cand_set) / len(job_set)) * 100)

# ==========================================
# 4. PYDANTIC SCHEMAS  (API response shapes)
# ==========================================
class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str

class JobOut(BaseModel):
    id: int
    title: str
    company: str
    location: str
    job_type: str
    salary_range: str
    description: str
    required_skills: str
    experience_level: str
    posted_by: int
    posted_at: str
    status: str

class ApplicationOut(BaseModel):
    id: int
    job_id: int
    candidate_id: int
    candidate_name: str
    candidate_email: str
    resume_path: Optional[str]
    cover_letter: str
    skills: str
    status: str
    match_score: str
    applied_at: str
    interview_date: Optional[str]
    interview_notes: Optional[str]

def job_to_dict(j: Job) -> dict:
    return {k: getattr(j, k) for k in JobOut.model_fields}

def app_to_dict(a: Application) -> dict:
    return {k: getattr(a, k) for k in ApplicationOut.model_fields}

def user_to_dict(u: User) -> dict:
    return {k: getattr(u, k) for k in UserOut.model_fields}

# ==========================================
# 5. HTML ROUTES  (unchanged behaviour)
# ==========================================

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html")

@app.post("/signup")
def signup_post(
    request: Request,
    name: str = Form(...), email: str = Form(...),
    password: str = Form(...), role: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.scalars(select(User).where(User.email == email)).first():
        return templates.TemplateResponse(request=request, name="signup.html",
                                          context={"error": "Email already registered."})
    new_user = User(name=name, email=email,
                    hashed_password=get_password_hash(password), role=role)
    db.add(new_user); db.commit()
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie("access_token", create_access_token({"sub": new_user.email}), httponly=True)
    return resp

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.post("/login")
def login_post(request: Request, email: str = Form(...),
               password: str = Form(...), db: Session = Depends(get_db)):
    user = db.scalars(select(User).where(User.email == email)).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(request=request, name="login.html",
                                          context={"error": "Invalid email or password."})
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie("access_token", create_access_token({"sub": user.email}), httponly=True)
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("access_token")
    return resp

@app.get("/", response_class=HTMLResponse)
def home_page(request: Request, current_user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    jobs = db.scalars(select(Job).where(Job.status == "open")).all()
    if current_user.role == "recruiter":
        my_jobs = db.scalars(select(Job).where(Job.posted_by == current_user.id)).all()
        my_job_ids = [j.id for j in my_jobs]
        # All applications across the recruiter's jobs
        all_applications = db.scalars(
            select(Application).where(Application.job_id.in_(my_job_ids))
        ).all() if my_job_ids else []
        # Dict: job_id -> [applications]  — used by the template for per-card counts
        apps_by_job: dict = {}
        for a in all_applications:
            apps_by_job.setdefault(a.job_id, []).append(a)
        return templates.TemplateResponse(request=request, name="index.html",
                                          context={"current_user": current_user,
                                                   "jobs": jobs, "my_jobs": my_jobs,
                                                   "all_applications": all_applications,
                                                   "apps_by_job": apps_by_job})
    else:
        my_applications = db.scalars(
            select(Application).where(Application.candidate_id == current_user.id)).all()
        applied_job_ids = {a.job_id for a in my_applications}
        return templates.TemplateResponse(request=request, name="index.html",
                                          context={"current_user": current_user, "jobs": jobs,
                                                   "my_applications": my_applications,
                                                   "applied_job_ids": applied_job_ids})

@app.get("/jobs/create", response_class=HTMLResponse)
def create_job_page(request: Request, current_user: User = Depends(get_current_user)):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request=request, name="create_job.html",
                                      context={"current_user": current_user})

@app.post("/jobs/create")
def create_job(
    title: str = Form(...), company: str = Form(...), location: str = Form(...),
    job_type: str = Form(...), salary_range: str = Form(...),
    description: str = Form(...), required_skills: str = Form(...),
    experience_level: str = Form(...),
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    db.add(Job(title=title, company=company, location=location, job_type=job_type,
               salary_range=salary_range, description=description,
               required_skills=required_skills, experience_level=experience_level,
               posted_by=current_user.id,
               posted_at=datetime.now().strftime("%Y-%m-%d %H:%M")))
    db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/jobs/update/{job_id}", response_class=HTMLResponse)
def update_job_page(request: Request, job_id: int,
                    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    job = db.get(Job, job_id)
    return templates.TemplateResponse(request=request, name="update_job.html",
                                      context={"job": job, "current_user": current_user})

@app.post("/jobs/update/{job_id}")
def update_job(
    job_id: int, title: str = Form(...), company: str = Form(...),
    location: str = Form(...), job_type: str = Form(...),
    salary_range: str = Form(...), description: str = Form(...),
    required_skills: str = Form(...), experience_level: str = Form(...),
    status: str = Form(...),
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    job = db.get(Job, job_id)
    if job:
        job.title = title; job.company = company; job.location = location
        job.job_type = job_type; job.salary_range = salary_range
        job.description = description; job.required_skills = required_skills
        job.experience_level = experience_level; job.status = status
        db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/jobs/delete/{job_id}")
def delete_job(job_id: int, current_user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    job = db.get(Job, job_id)
    if job:
        db.delete(job); db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/jobs/{job_id}/apply", response_class=HTMLResponse)
def apply_page(request: Request, job_id: int,
               current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or current_user.role != "candidate":
        return RedirectResponse(url="/", status_code=303)
    job = db.get(Job, job_id)
    return templates.TemplateResponse(request=request, name="apply.html",
                                      context={"job": job, "current_user": current_user})

@app.post("/jobs/{job_id}/apply")
async def apply_post(
    job_id: int, cover_letter: str = Form(...), skills: str = Form(...),
    resume: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if not current_user or current_user.role != "candidate":
        return RedirectResponse(url="/", status_code=303)
    job = db.get(Job, job_id)
    resume_path = None
    if resume and resume.filename:
        ext = os.path.splitext(resume.filename)[1]
        fname = f"resume_{current_user.id}_{job_id}_{datetime.now().timestamp()}{ext}"
        resume_path = f"uploads/{fname}"
        with open(os.path.join(UPLOAD_DIR, fname), "wb") as buf:
            shutil.copyfileobj(resume.file, buf)
    score = compute_match_score(job.required_skills if job else "", skills)
    db.add(Application(
        job_id=job_id, candidate_id=current_user.id,
        candidate_name=current_user.name, candidate_email=current_user.email,
        resume_path=resume_path, cover_letter=cover_letter, skills=skills,
        match_score=str(score), applied_at=datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/applications/{app_id}/update", response_class=HTMLResponse)
def update_application_page(request: Request, app_id: int,
                             current_user: User = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    application = db.get(Application, app_id)
    job = db.get(Job, application.job_id) if application else None
    return templates.TemplateResponse(request=request, name="update_application.html",
                                      context={"application": application, "job": job,
                                               "current_user": current_user})

@app.post("/applications/{app_id}/update")
def update_application(
    app_id: int, status: str = Form(...),
    interview_date: Optional[str] = Form(None),
    interview_notes: Optional[str] = Form(None),
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if not current_user or current_user.role != "recruiter":
        return RedirectResponse(url="/", status_code=303)
    application = db.get(Application, app_id)
    if application:
        application.status = status
        application.interview_date = interview_date
        application.interview_notes = interview_notes
        db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/applications/{app_id}/delete")
def delete_application(app_id: int, current_user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    application = db.get(Application, app_id)
    if application:
        if application.resume_path:
            old = os.path.join("static", application.resume_path)
            if os.path.exists(old):
                os.remove(old)
        db.delete(application); db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int,
               current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    job = db.get(Job, job_id)
    applications = db.scalars(select(Application).where(Application.job_id == job_id)).all()
    return templates.TemplateResponse(request=request, name="job_detail.html",
                                      context={"job": job, "applications": applications,
                                               "current_user": current_user})

# ==========================================
# 6. JSON API ROUTER  (/api/v1/...)
#    Auth: Authorization: Bearer <token>
# ==========================================
api = APIRouter(prefix="/api/v1", tags=["API"])

# ── Auth ──────────────────────────────────────────────────────────────────────

@api.post("/auth/signup", summary="Register a new user")
def api_signup(
    name: str = Form(...), email: str = Form(...),
    password: str = Form(...), role: str = Form(...),
    db: Session = Depends(get_db)
):
    """Returns a JWT token on success. role must be 'candidate' or 'recruiter'."""
    if db.scalars(select(User).where(User.email == email)).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(name=name, email=email,
                hashed_password=get_password_hash(password), role=role)
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer", "user": user_to_dict(user)}

@api.post("/auth/login", summary="Login and receive a JWT token")
def api_login(
    email: str = Form(...), password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Send email + password as form data, receive a Bearer token."""
    user = db.scalars(select(User).where(User.email == email)).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer", "user": user_to_dict(user)}

@api.get("/auth/me", summary="Get current authenticated user")
def api_me(current_user: User = Depends(get_api_user)):
    return user_to_dict(current_user)

# ── Jobs ─────────────────────────────────────────────────────────────────────

@api.get("/jobs", summary="List all open jobs")
def api_list_jobs(db: Session = Depends(get_db),
                  current_user: User = Depends(get_api_user)):
    jobs = db.scalars(select(Job).where(Job.status == "open")).all()
    return {"jobs": [job_to_dict(j) for j in jobs], "total": len(jobs)}

@api.get("/jobs/{job_id}", summary="Get a single job by ID")
def api_get_job(job_id: int, db: Session = Depends(get_db),
                current_user: User = Depends(get_api_user)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_dict(job)

@api.post("/jobs", summary="Create a job listing (recruiter only)")
def api_create_job(
    title: str = Form(...), company: str = Form(...), location: str = Form(...),
    job_type: str = Form(...), salary_range: str = Form(...),
    description: str = Form(...), required_skills: str = Form(...),
    experience_level: str = Form(...),
    db: Session = Depends(get_db), current_user: User = Depends(get_api_user)
):
    if current_user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Only recruiters can post jobs")
    job = Job(
        title=title, company=company, location=location, job_type=job_type,
        salary_range=salary_range, description=description,
        required_skills=required_skills, experience_level=experience_level,
        posted_by=current_user.id,
        posted_at=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    db.add(job); db.commit(); db.refresh(job)
    return {"message": "Job created", "job": job_to_dict(job)}

@api.put("/jobs/{job_id}", summary="Update a job listing (recruiter only)")
def api_update_job(
    job_id: int,
    title: str = Form(...), company: str = Form(...), location: str = Form(...),
    job_type: str = Form(...), salary_range: str = Form(...),
    description: str = Form(...), required_skills: str = Form(...),
    experience_level: str = Form(...), status: str = Form(...),
    db: Session = Depends(get_db), current_user: User = Depends(get_api_user)
):
    if current_user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Recruiters only")
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.title = title; job.company = company; job.location = location
    job.job_type = job_type; job.salary_range = salary_range
    job.description = description; job.required_skills = required_skills
    job.experience_level = experience_level; job.status = status
    db.commit()
    return {"message": "Job updated", "job": job_to_dict(job)}

@api.delete("/jobs/{job_id}", summary="Delete a job (recruiter only)")
def api_delete_job(job_id: int, db: Session = Depends(get_db),
                   current_user: User = Depends(get_api_user)):
    if current_user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Recruiters only")
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job); db.commit()
    return {"message": f"Job {job_id} deleted"}

# ── Applications ──────────────────────────────────────────────────────────────

@api.get("/jobs/{job_id}/applications", summary="List applications for a job (recruiter only)")
def api_job_applications(job_id: int, db: Session = Depends(get_db),
                          current_user: User = Depends(get_api_user)):
    if current_user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Recruiters only")
    apps = db.scalars(select(Application).where(Application.job_id == job_id)).all()
    return {"applications": [app_to_dict(a) for a in apps], "total": len(apps)}

@api.post("/jobs/{job_id}/apply", summary="Apply for a job (candidate only)")
async def api_apply(
    job_id: int,
    cover_letter: str = Form(...),
    skills: str = Form(...),
    resume: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_api_user)
):
    if current_user.role != "candidate":
        raise HTTPException(status_code=403, detail="Candidates only")
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check for duplicate application
    existing = db.scalars(
        select(Application).where(
            Application.job_id == job_id,
            Application.candidate_id == current_user.id
        )
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Already applied to this job")

    resume_path = None
    if resume and resume.filename:
        ext = os.path.splitext(resume.filename)[1]
        fname = f"resume_{current_user.id}_{job_id}_{datetime.now().timestamp()}{ext}"
        resume_path = f"uploads/{fname}"
        with open(os.path.join(UPLOAD_DIR, fname), "wb") as buf:
            shutil.copyfileobj(resume.file, buf)

    score = compute_match_score(job.required_skills, skills)
    new_app = Application(
        job_id=job_id, candidate_id=current_user.id,
        candidate_name=current_user.name, candidate_email=current_user.email,
        resume_path=resume_path, cover_letter=cover_letter, skills=skills,
        match_score=str(score),
        applied_at=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    db.add(new_app); db.commit(); db.refresh(new_app)
    return {"message": "Application submitted", "match_score": score,
            "application": app_to_dict(new_app)}

@api.get("/applications/my", summary="Get my applications (candidate only)")
def api_my_applications(db: Session = Depends(get_db),
                         current_user: User = Depends(get_api_user)):
    if current_user.role != "candidate":
        raise HTTPException(status_code=403, detail="Candidates only")
    apps = db.scalars(
        select(Application).where(Application.candidate_id == current_user.id)).all()
    return {"applications": [app_to_dict(a) for a in apps], "total": len(apps)}

@api.put("/applications/{app_id}", summary="Update application status (recruiter only)")
def api_update_application(
    app_id: int,
    status: str = Form(...),
    interview_date: Optional[str] = Form(None),
    interview_notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_api_user)
):
    if current_user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Recruiters only")
    application = db.get(Application, app_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    valid_statuses = {"Applied", "Shortlisted", "Interview", "Hired", "Rejected"}
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")
    application.status = status
    application.interview_date = interview_date
    application.interview_notes = interview_notes
    db.commit()
    return {"message": "Application updated", "application": app_to_dict(application)}

@api.delete("/applications/{app_id}", summary="Delete an application")
def api_delete_application(app_id: int, db: Session = Depends(get_db),
                            current_user: User = Depends(get_api_user)):
    application = db.get(Application, app_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    # Candidates can only delete their own; recruiters can delete any
    if current_user.role == "candidate" and application.candidate_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot delete another candidate's application")
    if application.resume_path:
        old = os.path.join("static", application.resume_path)
        if os.path.exists(old):
            os.remove(old)
    db.delete(application); db.commit()
    return {"message": f"Application {app_id} deleted"}

# Mount the API router
app.include_router(api)