import pyaudio
import threading
import wave
import typer
from pydub import AudioSegment
import numpy as np

app = typer.Typer()


def list_audio_sources() -> list:
    try:
        p = pyaudio.PyAudio()
        info = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev["maxInputChannels"] > 0:
                info.append(f"Index {i}: {dev['name']}")
        p.terminate()
        return info
    except OSError:
        typer.echo(
            "Errore: Impossibile accedere ai dispositivi audio. "
            + "Assicurati di avere configurato correttamente la scheda audio nell'ambiente dove stai lanciando il codice"
        )
        return []


def show_sources() -> list:
    try:
        sources = list_audio_sources()
        if len(sources) == 0:
            typer.echo("Nessuna sorgente audio attualmente disponibile")
            return sources

        typer.echo("Trovate le seguenti sorgenti audio:")
        for source in sources:
            typer.echo(source)
        return sources
    except Exception as message:
        typer.echo(f"Errore: Impossibile visualizzare le sorgenti audio: {message} ")
        return []


def record_audio():
    show_sources()
    sources = typer.prompt("Inserisci gli indici delle sorgenti audio da registrare (separati da spazi)")
    sources = [int(s) for s in sources.split()]
    duration = typer.prompt("Durata della registrazione in secondi", type=int)
    output = typer.prompt("Nome del file di output")
    format = typer.prompt("Formato di output (wav o mp3)")

    recorder = AudioRecorder(sources)
    typer.echo(f"Registrazione in corso da {len(sources)} sorgenti per {duration} secondi...")
    recorder.record(duration)

    if format.lower() == "wav":
        recorder.save_wav(output)
    elif format.lower() == "mp3":
        recorder.save_mp3(output)
    else:
        typer.echo(f"Formato non supportato: {format}. Uso WAV come predefinito.")
        recorder.save_wav(output)

    typer.echo(f"Registrazione completata. File salvato come {output}")


class AudioRecorder:
    def __init__(self, sources):
        self.sources = sources
        self.frames = {source: [] for source in sources}
        self.is_recording = False

    def record(self, duration):
        self.is_recording = True
        threads = []
        for source in self.sources:
            thread = threading.Thread(target=self._record_source, args=(source, duration))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        self.is_recording = False

    def save_wav(self, filename):
        combined = np.sum([np.frombuffer(b"".join(self.frames[s]), dtype=np.int16) for s in self.sources], axis=0)
        combined = (combined / len(self.sources)).astype(np.int16)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(combined.tobytes())

    def save_mp3(self, filename):
        combined = np.sum([np.frombuffer(b"".join(self.frames[s]), dtype=np.int16) for s in self.sources], axis=0)
        combined = (combined / len(self.sources)).astype(np.int16)

        audio_segment = AudioSegment(combined.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        audio_segment.export(filename, format="mp3")

    def _record_source(self, source, duration):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=source,
            frames_per_buffer=1024,
        )

        for _ in range(0, int(44100 / 1024 * duration)):
            if not self.is_recording:
                break
            data = stream.read(1024)
            self.frames[source].append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    while True:
        typer.echo("\nMenu:")
        typer.echo("1. Mostra sorgenti audio disponibili")
        typer.echo("2. Registra audio")
        typer.echo("3. Esci")

        choice = typer.prompt("Scegli un'opzione", type=int)

        if choice == 1:
            show_sources()
        elif choice == 2:
            record_audio()
        elif choice == 3:
            typer.echo("Arrivederci!")
            break
        else:
            typer.echo("Opzione non valida. Riprova.")


if __name__ == "__main__":
    # # Uso
    # sources = [1, 2]  # Indici delle sorgenti audio
    # recorder = AudioRecorder(sources)
    # recorder.record(duration=10)  # Registra per 10 secondi
    # recorder.save_wav("output.wav")
    typer.run(main)
