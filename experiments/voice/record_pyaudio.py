import io
import threading
import time
import pyaudio
import wave
import typer
from pydub import AudioSegment
import sounddevice as sd
import numpy as np

app = typer.Typer()


def list_audio_sources() -> list:
    try:
        p = pyaudio.PyAudio()
        info = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev["maxInputChannels"] > 0:
                info.append((i, dev["name"], "Input"))
            if dev["maxOutputChannels"] > 0:
                info.append((i, dev["name"], "Output"))
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
        for i, name, source_type in sources:
            typer.echo(f"Index {i}: {name} ({source_type})")
        return sources
    except Exception as message:
        typer.echo(f"Errore: Impossibile visualizzare le sorgenti audio: {message} ")
        return []


def countdown(duration, recorder):
    for remaining in range(duration, 0, -1):
        if not recorder.is_recording:
            break
        typer.echo(f"\rTempo rimanente: {remaining} secondi", nl=False)
        time.sleep(1)
    typer.echo("\nRegistrazione completata!")


def record_audio():
    show_sources()
    mic_index = typer.prompt("Inserisci l'indice del microfono (input)", type=int)
    system_index = typer.prompt("Inserisci l'indice dell'audio di sistema (output)", type=int)
    duration = typer.prompt("Durata della registrazione in secondi", type=int)
    format = typer.prompt("Formato di output (wav o mp3)")
    output = typer.prompt("Nome del file di output")

    recorder = AudioRecorder(mic_index, system_index)
    recorder.print_device_info()  # Stampa le informazioni sui dispositivi
    typer.echo(f"Registrazione in corso per {duration} secondi...")

    # Avvia il thread del countdown
    countdown_thread = threading.Thread(target=countdown, args=(duration, recorder))
    countdown_thread.start()

    # Avvia la registrazione
    recorder.record(duration)

    # Attendi che il thread del countdown termini
    countdown_thread.join()

    if format.lower() == "wav":
        recorder.save_wav(output)
    elif format.lower() == "mp3":
        recorder.save_mp3(output)
    else:
        typer.echo(f"Formato non supportato: {format}. Uso WAV come predefinito.")
        recorder.save_wav(output)

    typer.echo(f"File salvato come {output}")


class AudioRecorder:
    def __init__(self, mic_index, system_index):
        self.mic_index = mic_index
        self.system_index = system_index
        self.recording = None
        self.is_recording = False
        self.samplerate = 44100

    def record(self, duration):
        mic_info = sd.query_devices(self.mic_index, "input")
        system_info = sd.query_devices(self.system_index, "output")

        # Usa il sample rate nativo del dispositivo di input
        self.samplerate = int(mic_info["default_samplerate"])

        channels_in = mic_info["max_input_channels"]
        channels_out = system_info["max_output_channels"]

        self.recording_input = []
        self.recording_output = []
        self.is_recording = True

        def input_callback(indata, frames, time, status):
            if status:
                print(f"Input status: {status}")
            self.recording_input.append(indata.copy())

        def output_callback(outdata, frames, time, status):
            if status:
                print(f"Output status: {status}")
            self.recording_output.append(outdata.copy())

        try:
            with sd.InputStream(
                device=self.mic_index,
                channels=channels_in,
                callback=input_callback,
                samplerate=self.samplerate,
                blocksize=1024,
                latency="high",
            ):
                with sd.OutputStream(
                    device=self.system_index,
                    channels=channels_out,
                    callback=output_callback,
                    samplerate=self.samplerate,
                    blocksize=1024,
                    latency="high",
                ):
                    sd.sleep(int(duration * 1000))
        except sd.PortAudioError as e:
            typer.echo(f"Errore durante l'apertura dello stream audio: {e}")
            self.is_recording = False
            return
        self.is_recording = False

    def save_wav(self, filename):
        if not self.recording_input or not self.recording_output:
            typer.echo("Nessuna registrazione disponibile.")
            return

        # Combina input e output
        combined_input = np.concatenate(self.recording_input)
        combined_output = np.concatenate(self.recording_output)

        # Assicurati che abbiano la stessa lunghezza
        min_length = min(combined_input.shape[0], combined_output.shape[0])
        combined = np.mean([combined_input[:min_length], combined_output[:min_length]], axis=0)

        combined = (combined * 32767).astype(np.int16)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(combined.tobytes())

    def save_mp3(self, filename):
        if not self.recording_input or not self.recording_output:
            typer.echo("Nessuna registrazione disponibile.")
            return

        # Combina input e output
        combined_input = np.concatenate(self.recording_input)
        combined_output = np.concatenate(self.recording_output)

        # Assicurati che abbiano la stessa lunghezza
        min_length = min(combined_input.shape[0], combined_output.shape[0])
        combined = np.mean([combined_input[:min_length], combined_output[:min_length]], axis=0)

        combined = (combined * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(combined.tobytes())

        buffer.seek(0)
        audio_segment = AudioSegment.from_wav(buffer)
        audio_segment.export(filename, format="mp3")

    def print_device_info(self):
        mic_info = sd.query_devices(self.mic_index, "input")
        system_info = sd.query_devices(self.system_index, "output")
        typer.echo(
            f"Mic device: {mic_info['name']}, channels: {mic_info['max_input_channels']}, default samplerate: {mic_info['default_samplerate']}"
        )
        typer.echo(
            f"System device: {system_info['name']}, channels: {system_info['max_output_channels']}, default samplerate: {system_info['default_samplerate']}"
        )


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
