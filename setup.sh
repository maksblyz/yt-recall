#!/bin/bash

echo "🚀 Setting up Active Recall Learning App..."

# Create backend .env file
echo "📝 Creating backend environment file..."
cat > backend/.env << EOF
# API Keys - Replace with your actual keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
EOF

echo "✅ Backend environment file created at backend/.env"
echo "⚠️  Please edit backend/.env and add your actual API keys:"
echo "   - Get OpenAI API key from: https://platform.openai.com/api-keys"
echo "   - Get Groq API key from: https://console.groq.com/keys"

echo ""
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the application:"
echo "1. Edit backend/.env with your API keys"
echo "2. Start the backend: cd backend && uvicorn main:app --reload"
echo "3. Start the frontend: cd frontend && npm start"
echo ""
echo "The app will be available at http://localhost:3000" 