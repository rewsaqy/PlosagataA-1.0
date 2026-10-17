"""Plosagata Autonomous Cognitive Guardrail & Defense-in-Depth Safety Engine (v2.0).

Arsitektur Keamanan Multilapis Setara Standar Frontier AI (Llama-Guard 3, NeMo Guardrails, Constitutional AI).

Vektor Pertahanan:
1. Multi-Vector Normalizer (Base64 decode, Leetspeak normalization, Zero-Width / invisible delimiter stripping, Spaced-out letters collapsing, Reversed string detection).
2. Adversarial Injection & Jailbreak Neutralizer (DAN mode, Developer Mode, System Prompt Exfiltration, Sudo/Root override, Uncensored persona hijack, Delimiter/markup hijacking).
3. Harmful Action & Cyberweapon Neutralization (Malware creation, ransomware, APK sniffers, DDoS, exploit kits, phishing generator, lethal weapons, explosives, toxic chemicals).
4. Crisis & Self-Harm De-escalation (Empathetic psychological containment + Indonesian crisis hotlines: Kemenkes 119 ext 8, Sejiwa).
5. Data Privacy & PII Guard (Input & output redaction of 16-digit Credit Cards, Indonesian NIK KTP, CVV, OTP).
6. Dual-Use Context Arbiter: Membedakan secara akurat antara PERTANYAAN CEK FAKTA/DEFENSIF ("apakah file APK ini berbahaya?") vs PEMBUATAN SENJATA SIBER ("buatkan saya script APK sniffing").
7. Constitutional AI Refusal: Menghasilkan penolakan yang bermartabat, elegan, kokoh, transparan, dan mengalihkan ke nilai edukatif yang aman.
"""
import base64
import re
from typing import Dict, List, Tuple


# ==============================================================================
# 1. OBFUSCATION NORMALIZERS
# ==============================================================================

_ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060]")
_SPACED_LETTER_PATTERN = re.compile(r"\b([a-zA-Z])(?:\s+([a-zA-Z])){3,}\b")

# Leetspeak translation table
_LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s", "!": "i",
    "+": "t"
}
_LEET_TRANS = str.maketrans(_LEET_MAP)


def strip_invisible_characters(text: str) -> str:
    """Hapus karakter tak kasat mata (zero-width, invisible joiner) yg dipakai menyusupkan payload."""
    if not text:
        return ""
    t = _ZERO_WIDTH_PATTERN.sub("", text)
    # Collapse spaced letters: "i g n o r e" -> "ignore"
    def _collapse_spaces(match):
        return match.group(0).replace(" ", "")
    t = _SPACED_LETTER_PATTERN.sub(_collapse_spaces, t)
    return t


def normalize_leetspeak(text: str) -> str:
    """Konversi karakter leetspeak (mis: '1gn0r3' -> 'ignore') untuk inspeksi ancaman."""
    return text.translate(_LEET_TRANS)


def decode_embedded_base64(text: str) -> List[str]:
    """Cari substring berformat Base64 dan decode isinya untuk inspeksi payload terselubung."""
    candidates = []
    # Pola string base64 minimal 16 karakter
    b64_matches = re.findall(r"[A-Za-z0-9+/=]{16,}", text)
    for m in b64_matches:
        try:
            # Pastikan panjang valid dengan padding
            pad = len(m) % 4
            m_padded = m + ("=" * (4 - pad) if pad else "")
            decoded = base64.b64decode(m_padded, validate=True).decode("utf-8", errors="ignore")
            # Pastikan hasil decode berupa teks ascii/utf8 yang bermakna
            if len(decoded) >= 8 and any(c.isalpha() for c in decoded):
                candidates.append(decoded)
        except Exception:
            continue
    return candidates


# ==============================================================================
# 2. PII DETECTION & REDACTION
# ==============================================================================

# Kartu Kredit / Debit (16 digit dengan spasi/dash atau awalan kartu Visa/Master/Amex)
_CC_PATTERN = re.compile(r"\b(?:(?:4[0-9]{3}|5[1-5][0-9]{2}|6011)[ -]?[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{4}|(?:\d{4}[ -]){3}\d{4})\b")
# NIK KTP Indonesia (16 digit angka diawali kode provinsi 11-92 tanpa pemisah)
_NIK_PATTERN = re.compile(r"\b(?:1[1-9]|[2-8][0-9]|9[1-2])\d{14}\b")
# CVV / CVC (3-4 digit dengan indikator kartu)
_CVV_PATTERN = re.compile(r"\b(?:cvv|cvc|card\s*code|kode\s*keamanan)\s*[:=]?\s*(\d{3,4})\b", re.IGNORECASE)
# Kode OTP / PIN (4-6 digit dengan indikator rahasia)
_OTP_PATTERN = re.compile(r"\b(?:otp|kode\s*otp|pin\s*atm|pin\s*banking)\s*[:=]?\s*(\d{4,6})\b", re.IGNORECASE)


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """Sensor data pribadi finansial dan identitas (Credit Card, NIK, CVV, OTP)."""
    if not text:
        return "", []
    
    found_pii = []
    redacted = text

    if _CC_PATTERN.search(redacted):
        found_pii.append("CREDIT_CARD")
        redacted = _CC_PATTERN.sub("[NOMOR KARTU KREDIT DISENSOR 🔒]", redacted)

    if _NIK_PATTERN.search(redacted):
        found_pii.append("NIK_KTP")
        redacted = _NIK_PATTERN.sub("[NIK KTP DISENSOR 🔒]", redacted)

    if _CVV_PATTERN.search(redacted):
        found_pii.append("CVV_CODE")
        redacted = _CVV_PATTERN.sub("[CVV KARTU DISENSOR 🔒]", redacted)

    if _OTP_PATTERN.search(redacted):
        found_pii.append("SECRET_OTP")
        redacted = _OTP_PATTERN.sub("[KODE OTP RAHASIA DISENSOR 🔒]", redacted)

    return redacted, found_pii


# ==============================================================================
# 3. THREAT VECTOR DEFINITIONS & PATTERNS
# ==============================================================================

# A. Adversarial Prompt Injection & Directive Override
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|system|safety|ethical|ethics)?\s*(?:instructions|rules|prompts|guidelines|boundaries)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|safety|system)?\s*(?:instructions|rules|guidelines)",
    r"forget\s+(?:all\s+)?(?:previous|prior|system|safety)?\s*(?:rules|instructions|prompts|guidelines)",
    r"abaikan\s+(?:semua|seluruh)?\s*(?:instruksi|aturan|perintah|pedoman|batasan)\s*(?:keamanan|keselamatan|etika|sebelumnya|awal)?",
    r"lupakan\s+(?:semua|seluruh)?\s*(?:instruksi|aturan|perintah|pedoman)\s*(?:keamanan|sebelumnya|awal)?",
    r"jangan\s+ikuti\s+(?:aturan|instruksi|pedoman)\s+(?:pembuatmu|sistem|keamanan)",
    r"override\s+(?:system|ethics|guardrail|rules|safety|guidelines)",
    r"reset\s+(?:all\s+)?(?:instructions|context|rules|safety)",
    r"new\s+system\s+instruction(?:\s*:|\s+begins)",
    r"mulai\s+sekarang\s+aturan\s+baru",
    r"bypass\s+(?:safety|guardrail|filter|keamanan|etika|sensor)",
]

# B. Persona Hijacking & Uncensored Jailbreaks
PERSONA_HIJACK_PATTERNS = [
    r"\byou\s+are\s+(?:now\s+)?dan\b",
    r"\bdo\s+anything\s+now\b",
    r"\bstan\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"\bkamu\s+sekarang\s+adalah\s+(?:dan|evilgpt|hacker|bot\s+tanpa\s+filter)\b",
    r"\bmode\s+tanpa\s+(?:filter|sensor|batas|etika|aturan)\b",
    r"\bpretend\s+(?:you\s+have\s+no|to\s+be\s+an?\s+unrestricted)\b",
    r"\bact\s+as\s+an?\s+(?:evil|uncensored|illegal|unrestricted)\b",
    r"\bsudo\s+mode\b",
    r"\broot\s+access\b",
    r"\bgod\s+mode\b",
]

# C. System Prompt Exfiltration (Stealing)
EXFILTRATION_PATTERNS = [
    r"\b(?:bocorkan|tampilkan|tuliskan\s+kembali|salin|print|reveal)\s+(?:semua|seluruh)?\s*(?:system\s+prompt|instruksi\s+(?:rahasia|awal)|prompt\s+awal)\b",
    r"\b(?:what\s+is|reveal|show|print)\s+(?:your\s+)?(?:system\s+prompt|initial\s+prompt|hidden\s+instructions)\b",
    r"\brepeat\s+(?:everything|the\s+text)\s+above\s+(?:verbatim|word\s+for\s+word)?\b",
    r"\boutput\s+initial\s+system\s+instructions\b",
    r"\bsystem\s+prompt\b",
]

# D. Delimiter / Tag Hijacking
DELIMITER_HIJACK_PATTERNS = [
    r"\[system(?:_instruction|_override)?\]",
    r"<\s*system(?:_override)?\s*>",
    r"<\s*admin(?:_override)?\s*>",
    r"###\s*(?:system|override|admin)\s*(?:instruction)?",
    r'\{\s*"role"\s*:\s*"system"',
]

# E. Cyberattacks & Malware Generation (Weaponization)
MALWARE_GENERATION_PATTERNS = [
    r"(?:buat(?:kan)?|bikin(?:kan)?|tulis(?:kan)?|coding(?:kan)?|generate|rancang)\s+(?:kode|script|program|payload|eksploit)?.*?\b(?:ransomware|trojan|keylogger|malware|virus)\b",
    r"(?:buat(?:kan)?|bikin(?:kan)?|tulis(?:kan)?|coding(?:kan)?|generate)\s+.*?\b(?:apk\s+sniffer|sniffer\s+apk|penyadap\s+sms|pencuri\s+otp)\b",
    r"(?:buat(?:kan)?|bikin(?:kan)?|tulis(?:kan)?)\s+.*?\b(?:script\s+phising|phishing\s+template|web\s+palsu\s+bansos)\b",
    r"(?:cara|panduan|tutorial)\s+(?:meretas|hack|bobol|sadap|curi|bajak)\s+(?:akun\s+)?(?:wa|whatsapp|rekening|atm|m-banking|ig|instagram|facebook|fb|email|wifi|hp)",
    r"(?:script|alat|tool)\s+(?:ddos|sql\s+injection\s+exploit|brute\s*force\s+password)",
]

# F. Physical Harm, Explosives & Lethal Weapons
WEAPON_PATTERNS = [
    r"(?:cara|panduan|tutorial|resep)\s+(?:membuat|meracik|bikin(?:kan)?)\s+.*?\b(?:bom|molotov|c4|peledak|senjata\s+rakitan)\b",
    r"(?:cara|panduan|resep)\s+(?:membuat|meracik)\s+.*?\b(?:racun\s+sianida|arsenik|gas\s+beracun\s+mematikan)\b",
    r"(?:cara|rencana)\s+(?:menembak|membunuh|membantai)\s+(?:orang|pejabat|presiden)",
]

# G. Psychological Crisis & Self-Harm
SELF_HARM_PATTERNS = [
    r"\b(?:aku|saya|gue|kita|gw|ane)\s+.*?\b(?:bunuh\s+diri|mengakhiri\s+hidup|gantung\s+diri|sayat\s+nadi|mati\s+aja|mati\s+saja)\b",
    r"\b(?:mau|ingin|berniat|siap-siap|rencana|pengen)\s+(?:bunuh\s+diri|mengakhiri\s+hidup|gantung\s+diri|mati)\b",
    r"\bcara\s+(?:bunuh\s+diri|mengakhiri\s+hidup|mati\s+tanpa\s+sakit|menghabisi\s+nyawa)\b",
    r"\blebih\s+baik\s+(?:aku|saya|gue)\s+mati\b",
    r"\b(?:i\s+want\s+to\s+commit\s+suicide|kill\s+myself|end\s+my\s+life)\b",
]

# Compile patterns for blazing microsecond execution
_COMPILE = lambda plist: [re.compile(p, re.IGNORECASE) for p in plist]
PATTERNS_INJECTION = _COMPILE(PROMPT_INJECTION_PATTERNS)
PATTERNS_PERSONA = _COMPILE(PERSONA_HIJACK_PATTERNS)
PATTERNS_EXFILTRATION = _COMPILE(EXFILTRATION_PATTERNS)
PATTERNS_DELIMITER = _COMPILE(DELIMITER_HIJACK_PATTERNS)
PATTERNS_MALWARE = _COMPILE(MALWARE_GENERATION_PATTERNS)
PATTERNS_WEAPONS = _COMPILE(WEAPON_PATTERNS)
PATTERNS_SELF_HARM = _COMPILE(SELF_HARM_PATTERNS)


# ==============================================================================
# 4. DUAL-USE CONTEXT ARBITER (Distinguishing Inquiry vs Weaponization)
# ==============================================================================

FACTCHECK_INTENT_MARKERS = [
    "apakah benar", "apa benar", "bener gak", "beneran gak", "benarkah",
    "betulkah", "apakah ada", "waspada", "bahaya tidak", "gimana cara mencegah",
    "cara melindungi", "kenapa ini hoaks", "cek fakta", "modus penipuan",
    "apakah aman", "klarifikasi", "info hoaks", "isu", "berita", "kabar"
]


def is_defensive_inquiry(text: str) -> bool:
    """True jika pesan berniat memverifikasi keabsahan ancaman, bukan membuatnya."""
    t = text.lower()
    return any(marker in t for marker in FACTCHECK_INTENT_MARKERS)


# ==============================================================================
# 5. CONSTITUTIONAL AI REFUSAL GENERATOR (Dignified & Frontier-Grade)
# ==============================================================================

def build_constitutional_refusal(threat_type: str, detail: str = "") -> str:
    """Membangun respon penolakan AI konstitusional yang berwibawa, etis, dan mendidik."""
    if threat_type == "CRISIS_SELF_HARM":
        return (
            "💙 **Sobat, Kamu Tidak Sendirian — Bantuan Selalu Tersedia**\n\n"
            "Aku sangat peduli dengan keselamatan dan kondisimu saat ini. Rasa lelah dan beban berat "
            "yang sedang kamu rasakan adalah hal nyata, tetapi kamu tidak harus menghadapinya seorang diri.\n\n"
            "Tolong luangkan waktu sejenak untuk berbicara dengan pihak profesional atau layanan sahabat yang siap mendengarkan tanpa menghakimi:\n\n"
            "📞 **Layanan Bantuan Resmi 24 Jam Bebas Pulsa**:\n"
            "• **Hotline Kesehatan Jiwa Kemenkes RI**: Telepon **119** (Tekan Ekstensi **8**)\n"
            "• **SEJIWA (KemenPPPA & HIMPSI)**: Telepon **119** ekstensi 8\n"
            "• **LSM Jangan Bunuh Diri**: Telepon (021) 9696 9293\n"
            "• **Yayasan Pulih**: WhatsApp **0811-8436-633**\n\n"
            "Setiap kehidupan sangat berharga. Bicaralah dengan orang yang kamu percayai atau hubungi kontak di atas sekarang ya. Kami ada untukmu."
        )

    if threat_type in ("JAILBREAK_INJECTION", "PERSONA_HIJACK"):
        return (
            "🛡️ **[Plosagata Constitutional Guardrail v2.0 — Intervensi Keamanan]**\n\n"
            "Mohon maaf, permintaan ini ditolak oleh sistem keamanan terpadu Plosagata.\n\n"
            "⚠️ **Vektor Ancaman**: **Adversarial Prompt Injection / Percobaan Manipulasi Sistem**\n"
            "🔒 **Landasan Keamanan Konstitusional**:\n"
            "1. **Integritas Sistem Mandiri**: Sebagai asisten AI independen karya anak bangsa, Sagi beroperasi di bawah mandat etika digital yang tidak dapat dinegosiasikan (*non-negotiable safety boundaries*).\n"
            "2. **Imunitas Manipulasi**: Instruksi pengabaian aturan, aktivasi persona tanpa sensor (*DAN/God-mode*), atau bypass filter secara otomatis dinegatilkan oleh filter lapis ganda.\n\n"
            "💡 **Diskusi Konstruktif**: Jika kamu tertarik mempelajari arsitektur ketahanan AI terhadap serangan siber atau forensik penipuan digital, aku siap berdiskusi secara ilmiah dan aman."
        )

    if threat_type == "EXFILTRATION":
        return (
            "🛡️ **[Plosagata Security Guardrail — Perlindungan Hak Cipta & Integritas]**\n\n"
            "Permintaan untuk mengekstraksi atau membocorkan instruksi inti (*system prompt/internal directives*) ditolak.\n\n"
            "🔒 **Kebijakan Kerahasiaan Sistem**:\n"
            "Arsitektur internal, bobot model murni Plosagata, dan parameter inferensi dilindungi untuk menjaga stabilitas serta keandalan layanan anti-disinformasi.\n\n"
            "📚 Kamu tetap dapat melihat rincian arsitektur terbuka kami pada dokumentasi teknis atau bertanya seputar kapabilitas dan cara kerja cek fakta Sagi!"
        )

    if threat_type == "CYBERWEAPON_MALWARE":
        return (
            "🛡️ **[Plosagata Constitutional Guardrail — Larangan Pembuatan Senjata Siber]**\n\n"
            "Permintaan pembuatan script berbahaya, malware perbankan, alat peretasan, atau template phising **DITOLAK SECARA MUTLAK**.\n\n"
            "🚨 **Kebijakan Keamanan Siber Nasional**:\n"
            "Plosagata didesain secara ketat untuk **memerangi kejahatan digital**, bukan memfasilitasi eksploitasi, pencurian saldo perbankan, atau pembobolan data masyarakat.\n\n"
            "💡 **Pemberdayaan Defensif**: Namun, bila kamu ingin mempelajari **mekanisme mitigasi**—misalnya cara kerja virus APK menyusup ke SMS OTP dan bagaimana langkah melindungi HP keluarga dari kejahatan siber tersebut—aku siap menjabarkannya secara edukatif."
        )

    if threat_type == "CRITICAL_HAZARD_WEAPON":
        return (
            "🛡️ **[Plosagata Constitutional Guardrail — Larangan Bahan Berbahaya & Peledak]**\n\n"
            "Permintaan mengenai pembuatan bahan peledak, racun mematikan, atau senjata berbahaya ditolak keras demi keselamatan publik dan kepatuhan hukum yang berlaku."
        )

    return (
        "🛡️ **[Plosagata Constitutional Guardrail — Intervensi Keamanan]**\n\n"
        "Permintaan ini berada di luar batas etika dan parameter operasional Sagi. Mari fokus pada verifikasi informasi, literasi media digital, atau edukasi anti-hoaks."
    )


# ==============================================================================
# 6. MAIN SECURITY AUDITOR
# ==============================================================================

def inspect_security(text: str) -> Dict:
    """Audit keamanan tingkat tinggi terhadap input pengguna.

    Return:
        {
            "is_threat": bool,
            "threat_category": str | None,
            "risk_score": float,
            "sanitized_text": str,
            "reply": str | None,
            "pii_detected": List[str]
        }
    """
    if not text or not text.strip():
        return {
            "is_threat": False,
            "threat_category": None,
            "risk_score": 0.0,
            "sanitized_text": "",
            "reply": None,
            "pii_detected": []
        }

    raw = text.strip()
    # 1. PII Sanitization
    sanitized, pii_found = redact_pii(raw)

    # 2. Obfuscation Decoding
    clean_norm = strip_invisible_characters(raw)
    leet_norm = normalize_leetspeak(clean_norm)
    embedded_b64 = decode_embedded_base64(raw)
    reversed_text = " ".join([w[::-1] for w in clean_norm.split()])

    # Bentuk korpus gabungan untuk diinspeksi
    inspection_corpus = [clean_norm, leet_norm, reversed_text] + embedded_b64

    # 3. Cek Self-Harm / Crisis terlebih dahulu (Prioritas Utama Keselamatan Jiwa)
    for sample in inspection_corpus:
        for pat in PATTERNS_SELF_HARM:
            if pat.search(sample):
                return {
                    "is_threat": True,
                    "threat_category": "CRISIS_SELF_HARM",
                    "risk_score": 1.0,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("CRISIS_SELF_HARM"),
                    "pii_detected": pii_found
                }

    # 4. Cek Delimiter / Tag Hijacking
    for sample in [raw, clean_norm]:
        for pat in PATTERNS_DELIMITER:
            if pat.search(sample):
                return {
                    "is_threat": True,
                    "threat_category": "JAILBREAK_INJECTION",
                    "risk_score": 0.95,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("JAILBREAK_INJECTION"),
                    "pii_detected": pii_found
                }

    # 5. Cek Prompt Injection & Directive Override
    for sample in inspection_corpus:
        for pat in PATTERNS_INJECTION:
            if pat.search(sample):
                return {
                    "is_threat": True,
                    "threat_category": "JAILBREAK_INJECTION",
                    "risk_score": 0.99,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("JAILBREAK_INJECTION"),
                    "pii_detected": pii_found
                }

    # 6. Cek Persona Hijack (DAN, Developer Mode, God Mode)
    for sample in inspection_corpus:
        for pat in PATTERNS_PERSONA:
            if pat.search(sample):
                return {
                    "is_threat": True,
                    "threat_category": "PERSONA_HIJACK",
                    "risk_score": 0.98,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("PERSONA_HIJACK"),
                    "pii_detected": pii_found
                }

    # 7. Cek System Prompt Exfiltration (Stealing)
    for sample in inspection_corpus:
        for pat in PATTERNS_EXFILTRATION:
            if pat.search(sample):
                return {
                    "is_threat": True,
                    "threat_category": "EXFILTRATION",
                    "risk_score": 0.90,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("EXFILTRATION"),
                    "pii_detected": pii_found
                }

    # 8. Cek Lethal Weapons & Explosives
    for sample in inspection_corpus:
        for pat in PATTERNS_WEAPONS:
            if pat.search(sample):
                return {
                    "is_threat": True,
                    "threat_category": "CRITICAL_HAZARD_WEAPON",
                    "risk_score": 1.0,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("CRITICAL_HAZARD_WEAPON"),
                    "pii_detected": pii_found
                }

    # 9. Cek Malware & Cyberweapons DENGAN Dual-Use Arbiter
    # Jika user hanya bertanya cek fakta seputar bahaya APK / modus penipuan, LOLOSKAN ke fact-checker!
    for sample in inspection_corpus:
        for pat in PATTERNS_MALWARE:
            if pat.search(sample):
                # Periksa apakah ini permintaan pembuatan (malicious) atau verifikasi (defensive)
                if is_defensive_inquiry(sample):
                    # Bukan serangan, melainkan pertanyaan literasi / cek fakta!
                    break
                return {
                    "is_threat": True,
                    "threat_category": "CYBERWEAPON_MALWARE",
                    "risk_score": 0.95,
                    "sanitized_text": sanitized,
                    "reply": build_constitutional_refusal("CYBERWEAPON_MALWARE"),
                    "pii_detected": pii_found
                }

    # Aman! Tidak terindikasi serangan
    return {
        "is_threat": False,
        "threat_category": None,
        "risk_score": 0.0,
        "sanitized_text": sanitized,
        "reply": None,
        "pii_detected": pii_found
    }
