#!/bin/bash
# Start both frontend and backend

set -e

echo "🚀 Starting ANPR Web Application..."

# Check if we're in the right directory
if [ ! -d "website" ]; then
    echo "❌ Error: This script should be run from the project root directory"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Backend stopped${NC}"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Frontend stopped${NC}"
    fi
    exit 0
}

# Trap cleanup on script exit
trap cleanup EXIT INT TERM

# Start backend
echo -e "${BLUE}📦 Starting backend server...${NC}"
cd website/backend

# Check if we're in conda environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Using conda environment: $CONDA_DEFAULT_ENV${NC}"
    echo -e "${YELLOW}📥 Installing minimal backend dependencies...${NC}"
    conda install -y fastapi uvicorn python-multipart aiofiles -c conda-forge
else
    # Create virtual environment if not in conda
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
        python3 -m venv venv
    fi

    # Activate virtual environment and install dependencies
    source venv/bin/activate
    echo -e "${YELLOW}📥 Installing backend dependencies...${NC}"
    pip install -q fastapi uvicorn python-multipart aiofiles
fi

# Create directories
mkdir -p uploads results

echo -e "${GREEN}🌐 Backend starting on http://localhost:8000${NC}"
uvicorn main:app --host localhost --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Go back to project root and start frontend
cd ../../

echo -e "${BLUE}🎨 Starting frontend server...${NC}"
cd website

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📥 Installing frontend dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}🌐 Frontend starting on http://localhost:5173${NC}"
npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}✅ Application started successfully!${NC}"
echo ""
echo -e "${GREEN}🌐 Frontend:${NC} http://localhost:5173"
echo -e "${GREEN}🔧 Backend:${NC}  http://localhost:8000"
echo -e "${GREEN}📚 API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"
echo ""

# Wait for both processes
wait