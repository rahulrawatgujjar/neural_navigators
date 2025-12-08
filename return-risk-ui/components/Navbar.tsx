import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="w-full bg-slate-900 text-white shadow-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        
        {/* Logo / Title */}
        <Link href="/" className="text-xl font-bold tracking-wide">
          Return Risk Predictor
        </Link>

        {/* Menu */}
        <div className="space-x-8 text-sm font-medium flex items-center">
          
          <Link href="/" className="hover:text-indigo-400 transition">
            Home
          </Link>

          <Link href="/about" className="hover:text-indigo-400 transition">
            About
          </Link>

          {/* ✅ GitHub Link */}
          <a
            href="https://github.com/rahulrawatgujjar/neural_navigators"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 transition text-white font-semibold"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}
