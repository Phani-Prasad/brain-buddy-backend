"""
AI Tutor Pro - Backend API
Full-featured AI tutoring platform using OpenAI
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
import tempfile
import aiofiles
from pathlib import Path
import rag_service
import content_safety
import auth
import progress

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    auth.init_db()
    progress.init_progress_db()
    print("✅ Database initialized")
    yield

app = FastAPI(title="Brain Buddy", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# In-memory storage (replace with database in production)
conversations = {}
user_progress = {}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    subject: Optional[str] = "General"
    grade_level: Optional[str] = "High School"
    language: Optional[str] = "en"
    user_id: Optional[int] = None  # for progress tracking


class LogActivityRequest(BaseModel):
    user_id: int
    activity_type: str
    subject: Optional[str] = "General"
    score: Optional[int] = None
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]
    resources: List[str]


class ExplanationRequest(BaseModel):
    topic: str
    grade_level: str
    detail_level: str = "medium"  # simple, medium, detailed
    language: Optional[str] = "en"


class PracticeRequest(BaseModel):
    subject: str
    topic: str
    difficulty: str
    num_questions: int = 5
    language: Optional[str] = "en"


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/")
def root():
    return {
        "message": "Brain Buddy API",
        "version": "1.0.0",
        "status": "running"
    }


# =============================================
# Progress Tracking Endpoints
# =============================================

@app.post("/api/progress/log")
def log_user_activity(request: LogActivityRequest):
    """Log a user activity event."""
    progress.log_activity(
        user_id=request.user_id,
        activity_type=request.activity_type,
        subject=request.subject or "General",
        score=request.score,
        metadata=request.metadata,
    )
    return {"status": "logged"}


@app.get("/api/progress/{user_id}")
def get_progress_dashboard(user_id: int):
    """Get aggregated progress stats for the dashboard."""
    stats = progress.get_user_stats(user_id)
    return stats


# =============================================
# Authentication Endpoints
# =============================================

@app.post("/api/auth/register")
def register(request: RegisterRequest):
    """Register a new user and return a JWT token."""
    if not request.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if not request.email.strip() or "@" not in request.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    user = auth.create_user(request.username.strip(), request.email.strip(), request.password)
    token = auth.create_access_token({"sub": str(user["id"])})
    return {"token": token, "user": user}


@app.post("/api/auth/login")
def login(request: LoginRequest):
    """Login with email and password, return JWT token."""
    user = auth.authenticate_user(request.email.strip(), request.password)
    token = auth.create_access_token({"sub": str(user["id"])})
    return {"token": token, "user": user}


@app.get("/api/auth/me")
def get_me(authorization: Optional[str] = None):
    """Get current user info from Bearer token."""
    from fastapi import Request
    return {"message": "Use Authorization header"}


@app.post("/api/auth/me")
async def get_me_post(request_body: dict):
    """Validate token and return user info."""
    token = request_body.get("token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    payload = auth.decode_token(token)
    user_id = int(payload.get("sub", 0))
    user = auth.get_user_by_id(user_id)
    return {"user": user}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for conversational tutoring"""
    
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please add your API key to the .env file.")
    
    # Get or create conversation history
    if request.session_id not in conversations:
        conversations[request.session_id] = []
    
    conversation = conversations[request.session_id]
    
    # Language instruction
    lang_instruction = ""
    if request.language == "te":
        lang_instruction = "IMPORTANT: Respond in Telugu language (తెలుగు). Use Telugu script for all explanations.\n\n"
    
    system_prompt = f"""{lang_instruction}{content_safety.SAFETY_GUARDRAIL}
You are an expert AI tutor specializing in {request.subject} for {request.grade_level} students.

Your teaching approach:
1. Be patient, encouraging, and supportive
2. Explain concepts clearly with examples
3. Ask guiding questions to help students think
4. Break down complex topics into simple steps
5. Provide real-world applications
6. Never just give answers - guide students to discover solutions
7. Adapt your language to the student's grade level

When a student asks a question:
- First, assess their understanding
- Explain the underlying concept
- Guide them step-by-step
- Encourage them to try solving
- Provide hints, not direct answers
"""
    
    # 🛡️ Content safety check
    safety_result = await content_safety.check_content_safety(request.message, client)
    if not safety_result["safe"]:
        return ChatResponse(
            response=safety_result["message"],
            suggestions=["Ask me about math 📐", "Ask me about science 🔬", "Ask me about English 📚"],
            resources=[]
        )
    
    # Add user message to conversation
    conversation.append({"role": "user", "content": request.message})
    
    # Call OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation[-10:]  # Keep last 10 messages for context
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        assistant_message = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": assistant_message})
        
        # Generate follow-up suggestions
        suggestions = generate_suggestions(request.message, request.subject)
        
        # Generate learning resources
        resources = generate_resources(request.subject, request.message)
        
        return ChatResponse(
            response=assistant_message,
            suggestions=suggestions,
            resources=resources
        )
        
    except Exception as e:
        print(f"ERROR in chat endpoint: {str(e)}")  # Log to console
        import traceback
        traceback.print_exc()  # Print full traceback
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint — returns tokens via Server-Sent Events."""

    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    # Content safety check (fast, non-streaming)
    safety_result = await content_safety.check_content_safety(request.message, client)
    if not safety_result["safe"]:
        async def safety_gen():
            msg = safety_result["message"]
            yield f"data: {msg}\n\n"
            meta = json.dumps({"suggestions": ["Ask me about math 📐", "Ask me about science 🔬", "Ask me about English 📚"], "resources": []})
            yield f"data: [DONE]:{meta}\n\n"
        return StreamingResponse(safety_gen(), media_type="text/event-stream")

    # Conversation history
    if request.session_id not in conversations:
        conversations[request.session_id] = []
    conversation = conversations[request.session_id]

    lang_instruction = ""
    if request.language == "te":
        lang_instruction = "IMPORTANT: Respond in Telugu language (తెలుగు). Use Telugu script for all explanations.\n\n"

    system_prompt = f"""{lang_instruction}{content_safety.SAFETY_GUARDRAIL}
You are an expert AI tutor specializing in {request.subject} for {request.grade_level} students.

Your teaching approach:
1. Be patient, encouraging, and supportive
2. Explain concepts clearly with examples
3. Ask guiding questions to help students think
4. Break down complex topics into simple steps
5. Provide real-world applications
6. Never just give answers - guide students to discover solutions
7. Adapt your language to the student's grade level

When a student asks a question:
- First, assess their understanding
- Explain the underlying concept
- Guide them step-by-step
- Encourage them to try solving
- Provide hints, not direct answers
"""

    conversation.append({"role": "user", "content": request.message})

    async def generate():
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation[-10:]
                ],
                temperature=0.7,
                max_tokens=800,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    # Escape newlines so SSE stays single-line
                    escaped = delta.replace("\n", "\\n")
                    yield f"data: {escaped}\n\n"

            # Save assistant reply to history
            conversation.append({"role": "assistant", "content": full_response})

            # Send final metadata chunk
            suggestions = generate_suggestions(request.message, request.subject)
            resources = generate_resources(request.subject, request.message)
            meta = json.dumps({"suggestions": suggestions, "resources": resources})
            yield f"data: [DONE]:{meta}\n\n"

        except Exception as e:
            yield f"data: [ERROR]:{str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@app.post("/api/explain")
async def explain_topic(request: ExplanationRequest):
    """Get detailed explanation of a topic"""
    
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please add your API key to the .env file.")
    
    detail_prompts = {
        "simple": "Explain this in very simple terms, as if teaching a beginner. Use analogies and everyday examples.",
        "medium": "Provide a clear explanation with examples and key points.",
        "detailed": "Provide a comprehensive, in-depth explanation with examples, applications, and advanced concepts."
    }
    
    lang_instruction = ""
    if request.language == "te":
        lang_instruction = "IMPORTANT: Provide the entire explanation in Telugu language (తెలుగు). Use Telugu script.\n\n"
    
    prompt = f"""{lang_instruction}{content_safety.SAFETY_GUARDRAIL}
Explain the topic: {request.topic}

Grade Level: {request.grade_level}
Detail Level: {detail_prompts[request.detail_level]}

Structure your explanation as:
1. Simple Definition
2. Key Concepts
3. Examples
4. Common Mistakes
5. Practice Tips
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        return {
            "explanation": response.choices[0].message.content,
            "topic": request.topic,
            "grade_level": request.grade_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")


@app.post("/api/practice")
async def generate_practice(request: PracticeRequest):
    """Generate practice questions"""
    
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please add your API key to the .env file.")
    
    lang_instruction = ""
    if request.language == "te":
        lang_instruction = "IMPORTANT: Generate all questions, options, hints, and explanations in Telugu language (తెలుగు). Use Telugu script.\n\n"
    
    prompt = f"""{lang_instruction}Generate {request.num_questions} practice questions for:

Subject: {request.subject}
Topic: {request.topic}
Difficulty: {request.difficulty}

For each question, provide:
1. The question
2. Multiple choice options (if applicable)
3. Hints (don't reveal the answer)
4. The correct answer (in a separate section)
5. Explanation of the solution

IMPORTANT: Return ONLY a valid JSON array, no markdown formatting or code blocks.

Format as JSON array with structure:
[{{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "hints": ["hint1", "hint2"],
    "answer": "...",
    "explanation": "..."
}}]
"""
    
    # 🛡️ Content safety check on the topic
    safety_result = await content_safety.check_content_safety(request.topic, client)
    if not safety_result["safe"]:
        return {
            "questions": [],
            "subject": request.subject,
            "topic": request.topic,
            "difficulty": request.difficulty,
            "blocked": True,
            "message": safety_result["message"]
        }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from markdown code blocks if present
        if content.startswith("```"):
            # Remove markdown code block formatting
            lines = content.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        # Try to parse JSON
        try:
            questions = json.loads(content)
            # Ensure it's a list
            if not isinstance(questions, list):
                print(f"Warning: Expected list but got {type(questions)}")
                questions = []
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Content received: {content[:200]}...")
            # Return empty array on parse failure
            questions = []
        
        return {
            "questions": questions,
            "subject": request.subject,
            "topic": request.topic,
            "difficulty": request.difficulty
        }
        
    except Exception as e:
        print(f"Error in generate_practice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


@app.post("/api/check-answer")
async def check_answer(question: str, student_answer: str, correct_answer: str):
    """Check student's answer and provide feedback"""
    
    prompt = f"""A student answered a question. Provide constructive feedback.

Question: {question}
Student's Answer: {student_answer}
Correct Answer: {correct_answer}

Provide:
1. Is the answer correct? (Yes/Partially/No)
2. What they got right
3. What needs improvement
4. Hints for better understanding
5. Encouragement

Be supportive and educational, not just marking right/wrong.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            "feedback": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/subjects")
def get_subjects():
    """Get available subjects"""
    return {
        "subjects": [
            {"id": "math", "name": "Mathematics", "icon": "📐"},
            {"id": "science", "name": "Science", "icon": "🔬"},
            {"id": "english", "name": "English", "icon": "📚"},
            {"id": "history", "name": "History", "icon": "🏛️"},
            {"id": "programming", "name": "Programming", "icon": "💻"},
            {"id": "physics", "name": "Physics", "icon": "⚛️"},
            {"id": "chemistry", "name": "Chemistry", "icon": "🧪"},
            {"id": "biology", "name": "Biology", "icon": "🧬"},
        ]
    }


@app.get("/api/progress/{session_id}")
def get_progress(session_id: str):
    """Get student progress"""
    
    if session_id not in conversations:
        return {"messages": 0, "topics_covered": []}
    
    messages = len(conversations[session_id])
    
    return {
        "session_id": session_id,
        "total_messages": messages,
        "topics_covered": extract_topics(conversations[session_id]),
        "last_activity": datetime.now().isoformat()
    }


@app.post("/api/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text using OpenAI Whisper"""
    
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    # Validate file type
    allowed_types = ["audio/webm", "audio/mp3", "audio/mpeg", "audio/wav", "audio/ogg"]
    if audio.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {audio.content_type}")
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"audio_{datetime.now().timestamp()}.webm"
    
    try:
        # Save uploaded file
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        # Transcribe using Whisper
        with open(temp_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        
        return {
            "text": transcript.text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/voice/speak")
async def text_to_speech(text: str):
    """Convert text to speech using OpenAI TTS"""
    
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Create temporary file for audio output
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"speech_{datetime.now().timestamp()}.mp3"
    
    try:
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text[:4096]  # TTS has a 4096 character limit
        )
        
        # Save to temporary file
        response.stream_to_file(temp_path)
        
        # Return the audio file
        return FileResponse(
            temp_path,
            media_type="audio/mpeg",
            filename="response.mp3",
            background=None  # We'll clean up manually
        )
        
    except Exception as e:
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.post("/api/voice/chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = "default",
    subject: str = "General",
    grade_level: str = "High School",
    language: str = "en"
):
    """Complete voice chat: transcribe, process, and respond with audio"""
    
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    temp_audio_in = None
    temp_audio_out = None
    
    try:
        # Step 1: Transcribe audio to text
        temp_dir = tempfile.gettempdir()
        temp_audio_in = Path(temp_dir) / f"voice_in_{datetime.now().timestamp()}.webm"
        temp_audio_wav = Path(temp_dir) / f"voice_in_{datetime.now().timestamp()}.wav"
        
        # Save the uploaded WebM file
        async with aiofiles.open(temp_audio_in, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        # Try transcription directly with WebM
        transcript_text = None
        whisper_lang = "te" if language == "te" else "en"
        try:
            with open(temp_audio_in, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=whisper_lang
                )
                transcript_text = transcript.text
        except Exception as transcribe_error:
            # If WebM fails, try converting to WAV with pydub
            try:
                from pydub import AudioSegment
                print("WebM transcription failed, trying WAV conversion...")
                audio_segment = AudioSegment.from_file(str(temp_audio_in), format="webm")
                audio_segment.export(str(temp_audio_wav), format="wav")
                
                with open(temp_audio_wav, 'rb') as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=whisper_lang
                    )
                    transcript_text = transcript.text
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Audio transcription failed. Make sure FFmpeg is installed. Error: {str(e)}"
                )
        
        user_text = transcript_text
        
        # Step 2: Process through chat logic
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversation = conversations[session_id]
        
        lang_instruction = ""
        if language == "te":
            lang_instruction = "IMPORTANT: Respond in Telugu language (తెలుగు). Use Telugu script for all explanations.\n\n"
        
        system_prompt = f"""{lang_instruction}{content_safety.SAFETY_GUARDRAIL}
You are an expert AI tutor specializing in {subject} for {grade_level} students.

Your teaching approach:
1. Be patient, encouraging, and supportive
2. Explain concepts clearly with examples
3. Ask guiding questions to help students think
4. Break down complex topics into simple steps
5. Provide real-world applications
6. Never just give answers - guide students to discover solutions
7. Adapt your language to the student's grade level
8. Keep responses concise for voice interaction (2-3 sentences max)

When a student asks a question:
- First, assess their understanding
- Explain the underlying concept
- Guide them step-by-step
- Encourage them to try solving
- Provide hints, not direct answers
"""
        
        # 🛡️ Content safety check on transcribed text
        safety_result = await content_safety.check_content_safety(user_text, client)
        if not safety_result["safe"]:
            return {
                "user_text": user_text,
                "assistant_text": safety_result["message"],
                "audio_url": None,
                "timestamp": datetime.now().isoformat(),
                "blocked": True
            }
        
        conversation.append({"role": "user", "content": user_text})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation[-10:]
            ],
            temperature=0.7,
            max_tokens=300  # Shorter for voice
        )
        
        assistant_text = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": assistant_text})
        
        # Step 3: Convert response to speech
        temp_audio_out = Path(temp_dir) / f"voice_out_{datetime.now().timestamp()}.mp3"
        
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=assistant_text[:4096]
        )
        
        speech_response.stream_to_file(temp_audio_out)
        
        # Return both text and audio
        return {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "audio_url": f"/api/voice/audio/{temp_audio_out.name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice chat error: {str(e)}")
    
    finally:
        # Clean up input audio files with proper error handling
        import time
        time.sleep(0.1)  # Small delay to ensure files are released
        
        if temp_audio_in and temp_audio_in.exists():
            try:
                temp_audio_in.unlink()
            except PermissionError:
                # File still in use, will be cleaned up later
                pass
        
        # Clean up WAV files
        for wav_file in Path(tempfile.gettempdir()).glob("voice_in_*.wav"):
            try:
                wav_file.unlink()
            except (PermissionError, OSError):
                # File still in use, skip
                pass


@app.get("/api/voice/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve temporary audio files"""
    temp_dir = Path(tempfile.gettempdir())
    file_path = temp_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )


# =============================================
# RAG - Study Material Endpoints
# =============================================

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, extract text, chunk, embed, and store."""
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    # Validate file type
    filename = file.filename or "unknown.txt"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    if ext not in ["pdf", "docx", "txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: PDF, DOCX, TXT"
        )
    
    # Save to temp file
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"upload_{datetime.now().timestamp()}.{ext}"
    
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Extract text
        text = rag_service.extract_text(str(temp_path), ext)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the file.")
        
        # Chunk the text
        chunks = rag_service.chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced no text chunks.")
        
        # Generate unique doc ID
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Embed and store
        doc_meta = rag_service.store_document(doc_id, filename, chunks, client)
        
        return {
            "message": "Document uploaded and processed successfully",
            "document": doc_meta
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except (PermissionError, OSError):
                pass


@app.get("/api/documents")
def get_documents():
    """List all uploaded documents."""
    docs = rag_service.list_documents()
    return {"documents": docs}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    """Delete a document and its embeddings."""
    deleted = rag_service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}


@app.post("/api/documents/{doc_id}/query")
async def query_document(doc_id: str, question: str = Form(...)):
    """Ask a question about a specific document."""
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    # 🛡️ Content safety check on the question
    safety_result = await content_safety.check_content_safety(question, client)
    if not safety_result["safe"]:
        return {"answer": safety_result["message"], "blocked": True}
    
    try:
        result = rag_service.query_document(doc_id, question, client)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/documents/{doc_id}/flashcards")
async def get_flashcards(doc_id: str, count: int = 10):
    """Generate flashcards from a document using AI."""
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    try:
        result = rag_service.generate_flashcards(doc_id, client, count=min(count, 20))
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error generating flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")


@app.post("/api/documents/{doc_id}/summarize")
async def summarize_doc(doc_id: str):
    """Generate a summary of a document."""
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        result = rag_service.summarize_document(doc_id, client)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error summarizing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/documents/{doc_id}/flashcards")
async def generate_doc_flashcards(doc_id: str):
    """Generate flashcards from a document."""
    if not client.api_key or client.api_key == "":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        result = rag_service.generate_flashcards(doc_id, client)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error generating flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def generate_suggestions(message: str, subject: str) -> List[str]:
    """Generate follow-up question suggestions"""
    suggestions = [
        f"Can you explain more about this concept?",
        f"Show me a similar example",
        f"What are common mistakes in {subject}?",
        f"Give me practice problems"
    ]
    return suggestions[:3]


def generate_resources(subject: str, message: str) -> List[str]:
    """Generate learning resource suggestions"""
    resources = [
        f"📖 Study guide for {subject}",
        f"🎥 Video tutorials",
        f"📝 Practice worksheets",
        f"🎯 Interactive exercises"
    ]
    return resources[:3]


def extract_topics(conversation: List[dict]) -> List[str]:
    """Extract topics from conversation"""
    # Simple implementation - can be enhanced with NLP
    topics = []
    for msg in conversation:
        if msg["role"] == "user" and len(msg["content"]) > 10:
            topics.append(msg["content"][:50] + "...")
    return topics[:5]


if __name__ == "__main__":
    import uvicorn
    auth.init_db()  # Ensure DB is ready
    print("\n" + "="*60)
    print("Brain Buddy - Backend Server")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
