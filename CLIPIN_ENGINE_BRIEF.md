# CLIPIN Engine — Brief de développement

> Fork : `xvbaton/openshorts` (amont : `mutonby/openshorts`)  
> Cible initiale : MacBook Pro M1 Pro, **16 Go de mémoire unifiée**  
> Priorité : **V1 performance et fiabilité** ; V2 gaming multimodal ; V3 automatisation.  
> Ce document décrit des objectifs et observations utilisateur, **pas un audit du code déjà réalisé**.

## Vision

Transformer OpenShorts en un moteur local, modulaire et fiable pour créer automatiquement des Shorts gaming à partir de vidéos longues, avec ou sans commentaire vocal. Conserver la base existante et accélérer le pipeline avant toute refonte.

## Constats sur le job témoin

Une vidéo Zelda d'environ 12 minutes transcrites (227 segments), job commencé à 21:26:15 :
- Détection de sous-titres : à 21:26:31, aucun sous-titre incrusté détecté.
- Transcription Whisper : progression à 25 % à 21:30:28, 50 % à 21:33:34, 75 % à 21:36:15 et 100 % à 21:38:59 ; environ **700 s / 11 min 40 s**.
- Analyse LLM démarrée à 21:39:02, **12 fenêtres de scoring** ; shortlist de 10 fenêtres à 21:59:05, soit **20 min 03 s** de première passe.
- Analyse détaillée et rendu : durée non connue.

Configuration communiquée dans `.env` : `LLM_BASE_URL=http://host.docker.internal:11434/v1`, `LLM_MODEL=qwen3.5:4b`, `LLM_SCORE_BATCH=3`, `LLM_TIMEOUT=600`, `WHISPER_MODEL=small`, `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE=int8`, `MAX_CONCURRENT_JOBS=1`.

**Incohérence à diagnostiquer :** les logs du job mentionnaient `Model: qwen3.5` et `ollama ps` montrait `qwen3.5:latest` (et non `:4b`), avec un contexte de 16384 et environ 9,4 Go annoncés. Il peut s'agir d'un conteneur créé avant la modification du `.env`, d'une surcharge applicative ou d'autre chose : ne pas conclure sans tracer les paramètres effectifs.

Autres observations : swap macOS déjà monté à ~12,6 Go lors de tests de modèles, pression mémoire élevée ; `openshorts-renderer` observé redémarrant en boucle avec exit code 1. Les causes ne sont pas établies.

## V1 — Phase 1 : audit **en lecture seule**

- [ ] Cartographier ingestion → détection sous-titres → transcription → fenêtres/scoring → détail → découpage/rendu → export.
- [ ] Localiser modules, API, workers, files d'attente, configuration Docker et dépendances, en citant fichiers et lignes.
- [ ] Tracer effectivement la sélection du modèle Ollama, la propagation du `.env`, les éventuels overrides et paramètres de requêtes (contexte, raisonnement, limite de sortie, timeout).
- [ ] Identifier pourquoi la transcription CPU dans Docker est si lente ; évaluer l'usage d'un service WhisperKit/whisper.cpp natif macOS, tout en gardant un fallback Linux/CPU.
- [ ] Examiner les appels LLM : batch, taille des prompts, nombre d'appels, génération JSON, mode thinking, temps et statistiques de tokens.
- [ ] Examiner la cause du redémarrage du renderer avec les logs.
- [ ] Inventorier cache, reprise sur erreur, sauvegarde des étapes intermédiaires, tests et contrôles existants.
- [ ] Remettre un rapport : **faits confirmés / hypothèses / plan priorisé avec impact, effort et risques**.
- [ ] **Ne modifier aucun fichier au cours de ce premier audit.**

## V1 — Phase 2 : corrections incrémentales (après audit)

### Configuration & mémoire
- [ ] Sélection explicite du modèle local par défaut, cible `qwen3.5:4b`, vérification de sa disponibilité, pas de substitution silencieuse.
- [ ] Journaliser le modèle réellement servi, contexte et métriques Ollama disponibles ; rendre contexte et limite de sortie configurables.
- [ ] Comparer des contextes 4096/8192/plus grands selon le prompt réel ; éviter de tronquer des transcriptions.
- [ ] Tester le mode sans raisonnement pour le scoring lorsque pris en charge.
- [ ] Éviter de garder simultanément plusieurs modèles volumineux en mémoire sur la machine 16 Go.

### Transcription
- [ ] Comparer Whisper `small` CPU/Docker à une transcription accélérée native Apple Silicon ; ne pas imposer macOS aux autres installations.
- [ ] Réutiliser SRT/VTT et transcriptions en cache ; identifier les médias par empreinte pour éviter le retraitement.
- [ ] Mesurer durée, vitesse et pression mémoire.

### Sélection LLM
- [ ] Réduire les appels coûteux par scoring groupé et JSON court, validé ; gérer les réponses invalides et les timeouts.
- [ ] Présélectionner moins de candidats avant l'analyse détaillée, supprimer les doublons et chevauchements.
- [ ] Réutiliser scoring et shortlist en cas de reprise, sans recalcul inutile.
- [ ] Comparer qualité des extraits et temps, pas uniquement le temps brut.

### Fiabilité & observabilité
- [ ] Corriger le crash du renderer après diagnostic.
- [ ] Mesurer par job durées de chaque étape, modèle, paramètres, nombre d'appels et tokens si disponibles, erreurs et reprises.
- [ ] Permettre de reprendre au dernier checkpoint cohérent après échec.

## Critères d'acceptation V1

- Benchmark avant/après sur **la même vidéo témoin**, avec les mêmes paramètres clairement indiqués.
- Modèle réellement servi conforme au modèle demandé.
- Aucun redémarrage intempestif des workers pendant le traitement de référence.
- Pression mémoire maîtrisée ; pas de swap massif ajouté par le nouveau pipeline.
- Qualité et lisibilité des clips au moins comparables au pipeline d'origine.
- **Objectif expérimental, non garanti :** vidéo de 15 minutes → Shorts exportés en moins de 10 minutes sur M1 Pro 16 Go.

## V2 — Intelligence multimodale gaming (hors V1)

Les scènes intéressantes de GTA 6 et des jeux vidéo peuvent être **silencieuses** : explosions, poursuites, combats ou situations inattendues. La sélection ne doit pas dépendre exclusivement du dialogue.

- Préfiltrage audio (parole, intensité, variations) et visuel (images clés, mouvements, changements de scène).
- Analyse multimodale **uniquement des segments présélectionnés**, afin de limiter le temps d'inférence.
- Profils de scoring par jeu/type d'événement (gameplay, combat, poursuite, narration, clip humoristique).
- Cadrage vertical orienté action/UI de jeu, non centré exclusivement sur les visages.
- Scores et justifications audités, comparaison à une sélection humaine.

## V3 — Production autonome CLIPIN (hors V1)

- Ingestion de sources vidéo autorisées et de fichiers locaux.
- Traitement par lots, déduplication, reprise et supervision des jobs.
- Exports 9:16, sous-titres, titres, descriptions par plateforme.
- Intégration avec les flux Google Drive et publication existants, journal des publications et erreurs.

## Contraintes

1. Conserver OpenShorts comme socle, respecter les licences et mentions d'attribution.
2. Modifications petites, testables et réversibles ; pas de réécriture massive sans justification.
3. Compatibilité Docker/Linux préservée, accélérations macOS optionnelles.
4. Pas de dépendance cloud payante obligatoire, ni secrets committés.
5. **Aucun changement sur l'installation OpenShorts actuellement en production/test sur le Mac.**
6. Développer dans une branche dédiée ; demander des mesures réelles avant de déclarer un gain.
7. Ne pas démarrer V2/V3 avant stabilisation de V1.

## Livrables

1. Cartographie de code et rapport de diagnostic avec références aux fichiers.
2. Plan V1 classé par priorité, gain attendu, coût et risques.
3. Corrections dans des commits/PR séparés et tests correspondants.
4. Protocole de benchmark reproductible et résultats avant/après.

## Première consigne pour Codex / Work

> Lis ce document et audite le dépôt **en lecture seule**. Cartographie le pipeline et identifie les causes plausibles des 700 s de Whisper, 20 minutes de scoring, désaccord entre `qwen3.5:4b` dans `.env` et `qwen3.5:latest` dans Ollama, ainsi que du renderer qui redémarre. Cite les fichiers et lignes pertinents. Sépare faits établis, hypothèses à vérifier et corrections prioritaires. Ne modifie rien lors de cet audit.
