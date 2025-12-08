export default function Footer() {
  return (
    <footer className="w-full bg-slate-900 text-white">
      <div className="max-w-7xl mx-auto px-6 py-4 text-center text-sm text-slate-300">
        © {new Date().getFullYear()} Return Risk Prediction System  
        <br />
        Built using Machine Learning, FastAPI & Next.js
      </div>
    </footer>
  );
}