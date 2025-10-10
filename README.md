# Morgan’s Code for Antarctic Ice Shelf Benchmark Dataset

## Dataset Information
**Dataset provided by:** C. Baumhoer, DLR  

This dataset supports benchmarking for Antarctic ice shelf analysis using data from multiple satellite sources.

---

## 📁 File Structure

```
ICE-BENCH/
│
├── Envisat/
│   ├── scenes/
│   ├── masks/
│   └── test_envisat/
│
├── ERS/
│   ├── scenes/
│   ├── masks/
│   └── test_ERS/
│
└── Sentinel-1/
    ├── scenes/
    ├── masks/
    └── test_s1/
```

---

## ⚙️ Using LaTeX on Jasmin

### 1. Install TeX Live
Run the installation script:

```bash
bash install-texlive.sh
```

### 2. Update Your PATH
After installation, add TeX Live to your PATH:

```bash
export PATH="$HOME/texlive/$(date +%Y)/bin/$(ls $HOME/texlive/$(date +%Y)/bin | head -n1):$PATH"
```

---
