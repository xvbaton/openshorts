# Transcription CLIPIN sur Apple Silicon

## Choix du moteur

| Moteur | Accélération M1 Pro | Intégration et timestamps |
|---|---|---|
| WhisperKit | Core ML, ici `cpuAndGPU` pour l’encodeur et le décodeur | Le CLI exporte le texte, les segments, les mots horodatés et les métriques internes dans un JSON. Un petit pont HTTP transmet ce rapport au backend. [CLI et rapports](https://github.com/argmaxinc/argmax-oss-swift/blob/v1.1.1/Sources/ArgmaxCLI/TranscribeCLI.swift#L497-L537) |
| whisper.cpp | Metal, avec option Core ML pour l’encodeur | Le serveur natif renvoie des timestamps de tokens dans son JSON détaillé. Il faudrait regrouper les tokens en mots et vérifier les bornes. [Accélération](https://github.com/ggml-org/whisper.cpp/tree/v1.9.5#core-ml-support) · [Format du serveur](https://github.com/ggml-org/whisper.cpp/blob/v1.9.5/examples/server/server.cpp#L1070-L1171) |
| faster-whisper | CPU dans Docker sur le Mac | Chemin existant et repli automatique. Les mots, segments et métadonnées gardent le format OpenShorts. |

Le pont écoute sur la boucle locale. Il traite une transcription à la fois, lance le CLI avec le modèle Core ML déjà présent sur disque et attend la fin du processus avant de répondre. Qwen peut démarrer après cette réponse. Le CLI charge l’audio par blocs pour limiter l’allocation audio. Chaque transcription paie la réinitialisation du modèle, à inclure dans la mesure. Une autre requête reçue pendant une transcription revient immédiatement sur CPU. Garder `MAX_CONCURRENT_JOBS=1` sur le Mac pour éviter le chevauchement de jobs indépendants.

Un dépassement de la limite serveur de 600 s arrête et récupère le processus avant la réponse d’erreur. Le client Docker attend jusqu’à 660 s. Une panne réseau ou un arrêt forcé du pont peut empêcher le client de confirmer la fin du processus natif. La mémoire unifiée, la pression mémoire et le swap restent à mesurer sur le Mac. Le pont accepte un WAV de 128 Mio maximum, soit environ 70 minutes en mono PCM 16 kHz, puis le backend revient sur CPU.

## Préparation du Mac

Dans ce dépôt, compiler le CLI du tag [Argmax v1.1.1](https://github.com/argmaxinc/argmax-oss-swift/tree/v1.1.1) avec `swift build -c release --product argmax-cli`. Il faut macOS 14 ou ultérieur et un outillage Swift compatible avec le [Package.swift du tag](https://github.com/argmaxinc/argmax-oss-swift/blob/v1.1.1/Package.swift). Préparer localement le dossier du modèle multilingue [openai_whisper-small](https://huggingface.co/argmaxinc/whisperkit-coreml/tree/main/openai_whisper-small) et les fichiers du tokenizer. Pour des mesures répétables, noter la révision exacte du modèle.

Exemple de lancement une fois le CLI et le modèle présents sous `.cache/`.

```sh
python3 whisperkit_host.py \
  --cli .cache/argmax-oss-swift/.build/release/argmax-cli \
  --model-path .cache/models/openai_whisper-small
```

Dans le profil Docker de CLIPIN, choisir `TRANSCRIBE_BACKEND=whisperkit` et `WHISPERKIT_URL=http://host.docker.internal:50060/transcribe`. `WHISPERKIT_MODEL=openai_whisper-small` vérifie le nom du dossier servi et permet la réutilisation du cache YouTube partagé pour ce modèle. Sur un backend Python exécuté directement sur le Mac, utiliser `http://127.0.0.1:50060/transcribe`. Le repli utilise `WHISPER_MODEL` sur CPU en int8.

## Protocole Zelda

1. Placer le fichier Zelda de référence dans `.cache/asr-benchmark/zelda.mp4` et vérifier son empreinte avec `shasum -a 256`. Conserver ce fichier pour tous les essais. Noter macOS, Docker Desktop, ses CPU et sa RAM, la version du CLI et les versions des modèles.
2. Préinstaller les poids des deux moteurs. Relever `memory_pressure` et `sysctl vm.swapusage` avant, pendant et après chaque passage. Garder un seul job, et relever les autres modèles déjà chargés sur le Mac. Le CLI quitte après chaque transcription, y compris pour les essais avec caches système chauds.
3. Lancer la transcription seule dans des conteneurs jetables. Utiliser un dossier de résultats neuf par moteur et répétition. Le tag d’image du benchmark est distinct de celui de l’installation en cours.

```sh
bench="$PWD/.cache/asr-benchmark"
mkdir -p "$bench/cpu-1" "$bench/native-1"
docker build -t clipin-asr-bench:local .
docker run --rm \
  --mount type=volume,source=clipin-asr-bench-hf,target=/app/.cache/huggingface \
  clipin-asr-bench:local python -c \
  'from faster_whisper.utils import download_model; download_model("small")'

run_asr() {
  /usr/bin/time -p docker run --rm \
    --mount "type=bind,source=$bench,target=/bench" \
    --mount type=volume,source=clipin-asr-bench-hf,target=/app/.cache/huggingface \
    --env PREPARE_TRANSCRIPT=1 --env ASR_HOST_SLOTS=0 \
    --env TRANSCRIBE_BACKEND="$1" \
    --env WHISPER_MODEL=small --env WHISPER_DEVICE=cpu --env WHISPER_COMPUTE=int8 \
    --env WHISPERKIT_MODEL=openai_whisper-small \
    --env WHISPERKIT_URL=http://host.docker.internal:50060/transcribe \
    clipin-asr-bench:local python -u main.py --prepare-only \
    -i /bench/zelda.mp4 -o "/bench/$2" || return
  test -s "$bench/$2/.transcript_checkpoint.json"
}

run_asr whisper cpu-1
run_asr whisperkit native-1
jq '.transcript.metadata' "$bench/native-1/.transcript_checkpoint.json"
```

Le volume Hugging Face est dédié au benchmark. Relever le temps total et, pour le natif, `wall_seconds` et les temps internes du rapport. Vérifier `metadata.backend == "whisperkit"` avant de compter un passage comme natif. Faire trois répétitions par moteur et prendre la médiane. Rejouer aussi une commande avec son dossier déjà rempli pour vérifier le chemin de reprise.

4. Produire une transcription humaine de référence. Calculer le taux d’erreur de mots `(substitutions + suppressions + insertions) / mots de référence`, avec la même normalisation pour les deux sorties (casse, ponctuation, espaces, accents conservés). Si l’annotation intégrale coûte trop cher, figer trois extraits continus de 60 s au début, au milieu et à la fin, et rendre le périmètre explicite.
5. Annoter au moins 30 mots répartis dans ces extraits, avec leur début et leur fin réels sur la forme d’onde. Inclure des débuts après silence, des fins de phrase et du gameplay sous la voix. Pour chaque moteur, renseigner la part de mots retrouvés, puis la médiane et le 90e percentile des erreurs absolues de début et de fin pour les mots alignés. Écouter cinq coupes aux bornes détectées et relever les mots tronqués.

## Résultats Zelda mesurés

MacBook Pro M1 Pro, 16 Gio, macOS 27.2.1. Docker 29.8.2, Linux arm64, 10 CPU et 7,75 Gio alloués. La copie de référence de 734,49 s correspond par SHA-256 aux deux fichiers de l’installation OpenShorts. WhisperKit utilise le CLI Argmax v1.1.1 et `openai_whisper-small` à la révision `0f63a780`. Les mesures d’origine concernent le commit `2948763`.

| Passage | Durée de bout en bout | Mémoire libre avant / min. / après | Swap utilisé avant → après |
|---|---:|---:|---:|
| WhisperKit 1, trace Core ML | 170,73 s | 38 % / 29 % / 50 % | 3 256 → 4 588 Mio |
| WhisperKit 2, trace Metal de 12 s | 156,12 s | 53 % / 34 % / 54 % | 4 564 → 4 578 Mio |
| WhisperKit 3, sans trace | 154,31 s | 54 % / 45 % / 56 % | 4 578 → 4 474 Mio |
| Reprise du JSON WhisperKit 1 | 4,51 s | 47 % / 47 % / 55 % | 4 580 → 4 564 Mio |
| faster-whisper CPU int8, passage complet | 931,48 s | 56 % / 28 % / 36 % | 4 474 → 3 377 Mio |

La médiane WhisperKit est de **156,12 s** de bout en bout et **150,18 s** dans le processus natif. Les trois passages ont produit un résultat `whisperkit`, sans repli CPU. La comparaison au passage CPU complet donne **5,97×** sur cette vidéo. Le compteur ASR CPU a indiqué 909 s. Le passage CPU 2 a été arrêté sur demande après le jalon de 50 % à 526 s. Aucun troisième passage CPU n’a démarré.

La première trace Core ML était inexploitable à l’export. La trace Metal du passage 2 montre **9 346 intervalles Compute actifs** attribués au PID `argmax-cli` sur le GPU M1 Pro et un pic d’allocation Metal de **998 Mio** sur l’échantillon de 12 s. Les trois PID natifs avaient disparu à la fin de leurs passages. Le pic de mémoire résidente échantillonnée était de **933 Mio**. Le swap et l’indicateur de mémoire libre sont des mesures système. Le premier passage instrumenté a ajouté **1 332 Mio de swap utilisé** au relevé après passage.

La reprise a conservé l’empreinte et la date de modification du fichier JSON, sans lancer de processus natif. Les trois JSON bruts contiennent le même texte, les mêmes 233 segments et les mêmes **2 988 mots horodatés**. L’analyse a relevé 10 chevauchements de frontières, dont un de 4,58 s entre la fin d’un mot et le début du suivant, et trois reculs du début d’un mot, jusqu’à 1,24 s.

Le rejeu de la normalisation sur les trois JSON a éliminé ces chevauchements et reculs en conservant tous les mots, leur ordre et leurs métadonnées. Les bornes modifiées conservent les valeurs brutes dans `raw_start` et `raw_end`. Les 11 mots dont les bornes étaient égales restent signalés par une durée nulle. La précision acoustique et le taux d’erreur de mots restent à mesurer sur une annotation humaine. Ce rejeu a utilisé uniquement les JSON enregistrés.

Les mesures, logs, JSON bruts, copie normalisée et trace Metal sont conservés dans `.cache/asr-benchmark/`. L’installation OpenShorts est restée intacte. Les conteneurs de benchmark étaient temporaires.

Le JSON versionné `.transcript_checkpoint.json` reste dans le dossier du job après la fin et suit la rétention de ce dossier. La reprise vérifie la source par SHA-256 et la structure des timestamps. Une copie du même fichier peut réutiliser le transcript. Un nouvel upload local créant un autre job a son propre dossier. Un ancien JSON sans empreinte entraîne un nouveau passage ASR. Pour comparer deux moteurs, utiliser deux dossiers, car un transcript valide reste réutilisable pour ce job même si le moteur configuré change.

Le transfert existant vers les workers distants conserve le transcript préparé par l’API. Un transcript calculé seulement sur un worker distant reste soumis au nettoyage de son dossier de travail.
