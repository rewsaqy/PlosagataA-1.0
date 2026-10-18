"""Memori sesi Sagi: topik terakhir + riwayat giliran percakapan (thread-safe).

Riwayat giliran dipakai untuk membangun prompt multi-turn (sagi.prompt);
topik dipakai logika follow-up (merujuk berita terakhir).
"""
import threading
import time

MAX_SESI = 500      # batas jumlah sesi in-memory (per proses)
SESI_UMUR = 7200    # detik — sesi lebih tua dari ini dibuang
MAKS_GILIRAN = 16   # riwayat giliran terakhir yang disimpan per sesi


class Sesi:
    __slots__ = ("topik", "giliran", "aktif")

    def __init__(self):
        self.topik = None    # berita terakhir yang ditempel (str|None)
        self.giliran = []    # [(pesan_user, balasan), ...] lama -> baru
        self.aktif = time.time()


class SesiStore:
    def __init__(self, maks_sesi: int = MAX_SESI, umur: int = SESI_UMUR):
        self.maks_sesi = int(maks_sesi)
        self.umur = int(umur)
        self._data: dict[str, Sesi] = {}
        self._lock = threading.Lock()

    def _pangkas(self, now: float):
        """Buang sesi kadaluarsa, lalu yang paling lama bila masih melebihi batas."""
        tua = [k for k, s in self._data.items() if now - s.aktif > self.umur]
        for k in tua:
            del self._data[k]
        while len(self._data) > self.maks_sesi:
            self._data.pop(next(iter(self._data)))

    def ambil(self, session_id: str):
        """Ambil/buat slot sesi + perbarui timer. None bila session_id kosong."""
        if not session_id:
            return None
        now = time.time()
        with self._lock:
            if len(self._data) > self.maks_sesi:
                self._pangkas(now)
            s = self._data.get(session_id)
            if s is None:
                s = Sesi()
                self._data[session_id] = s
            s.aktif = now
            return s

    def topik(self, session_id: str):
        """Topik berita aktif untuk sesi (utk /status)."""
        if not session_id:
            return None
        with self._lock:
            s = self._data.get(session_id)
            return s.topik if s else None

    def catat_giliran(self, session_id: str, pesan_user: str, balasan: str):
        """Simpan 1 giliran percakapan (balasan tanpa blok sitasi)."""
        if not session_id:
            return
        with self._lock:
            s = self._data.get(session_id)
            if s is None:
                s = Sesi()
                self._data[session_id] = s
            s.giliran.append((str(pesan_user or ""), str(balasan or "")))
            if len(s.giliran) > MAKS_GILIRAN:
                del s.giliran[: len(s.giliran) - MAKS_GILIRAN]
            s.aktif = time.time()

    def hapus(self, session_id: str):
        with self._lock:
            self._data.pop(session_id, None)

    def jumlah(self) -> int:
        with self._lock:
            return len(self._data)
