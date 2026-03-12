#!/bin/bash
# Start ANPR System - Backend + Frontend

set -e

PROJECT_DIR="/home/akhil/3-2"

echo "======================================"
echo "Starting ANPR System (v1.0)"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to handle cleanup
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    pkill -f "python.*main.py" || true
    pkill -f "npm.*run.*dev" || true
    sleep 1
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if ports are available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Start Backend
echo -e "${YELLOW}1. Starting Backend API...${NC}"
cd "$PROJECT_DIR/backend"
conda run -n ml_workspace python main.py > /tmp/anpr_backend.log 2>&1 &
BACKEND_PID=$!
echo -e "   Backend PID: $BACKEND_PID"

# Wait for backend to start
echo -e "   Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Backend failed to start${NC}"
    echo "   Check log: tail -50 /tmp/anpr_backend.log"
    exit 1
fi

# Test health endpoint
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Backend running on http://localhost:8000${NC}"
else
    echo -e "   ${RED}✗ Backend health check failed${NC}"
    kill $BACKEND_PID || true
    echo "   Check log: tail -50 /tmp/anpr_backend.log"
    exit 1
fi

# Start Frontend
echo -e "${YELLOW}2. Starting Frontend...${NC}"
cd "$PROJECT_DIR/website"
npm run dev > /tmp/anpr_frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "   Frontend PID: $FRONTEND_PID"

# Wait for frontend to start
echo -e "   Waiting for frontend to compile..."
sleep 8

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Frontend failed to start${NC}"
    echo "   Check log: tail -50 /tmp/anpr_frontend.log"
    kill $BACKEND_PID || true
    exit 1
fi

echo -e "   ${GREEN}✓ Frontend running on http://localhost:5173${NC}"

# Print summary
echo ""
echo "======================================"
echo -e "${GREEN}✓ ANPR System is Ready!${NC}"
echo "======================================"
echo ""
echo "Frontend:  http://localhost:5173"
echo "Backend:   http://localhost:8000"
echo "API Docs:  http://localhost:8000/api/docs"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/anpr_backend.log"
echo "  Frontend: tail -f /tmp/anpr_frontend.log"
echo ""
echo "Press Ctrl+C to stop all services..."
echo ""

# Keep running
wait
