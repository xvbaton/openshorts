import React, { useCallback, useEffect, useState } from 'react';
import { FolderOpen, Loader2, RefreshCw } from 'lucide-react';
import { apiFetch, apiJson } from '../lib/api';

// Self-host projects are read from the server's output directory, not browser
// storage. Closing the tab does not detach or restart the running job.
export default function LocalProjects({ onOpen }) {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [opening, setOpening] = useState(null);
  const refresh = useCallback(async () => {
    try {
      const data = await apiJson('/api/local/projects');
      setProjects(data.projects || []);
      setError('');
    } catch (e) {
      setError('Impossible de charger les projets locaux.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  const open = async (id) => {
    setOpening(id);
    try {
      await onOpen(id);
    } catch (e) {
      setError('Ce projet ne peut pas être ouvert pour le moment.');
    } finally {
      setOpening(null);
    }
  };

  const stop = async (id) => {
    if (!window.confirm('Arrêter cette génération ?')) return;
    try {
      const response = await apiFetch(`/api/local/projects/${id}/stop`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      await refresh();
    } catch (e) {
      setError('Impossible d’arrêter ce projet.');
    }
  };

  return (
    <div className="h-full overflow-y-auto p-5 sm:p-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between gap-3 mb-6">
        <div>
          <h1 className="font-display text-2xl text-ink">Mes projets</h1>
          <p className="text-muted text-sm mt-1">Les générations continuent même si tu fermes cette page.</p>
        </div>
        <button type="button" className="btn-ghost" onClick={refresh} aria-label="Actualiser">
          <RefreshCw size={16} /> Actualiser
        </button>
      </div>
      {error && <p className="text-danger text-sm mb-4">{error}</p>}
      {loading && <Loader2 className="animate-spin text-brass" />}
      {!loading && projects.length === 0 && <p className="text-muted">Aucun projet enregistré.</p>}
      <div className="space-y-3">
        {projects.map((project) => {
          const active = ['queued', 'processing'].includes(project.status);
          return (
            <div key={project.job_id} className="card p-4 flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-ink text-sm font-medium flex items-center gap-2"><FolderOpen size={16} /> Projet {project.job_id.slice(0, 8)}</p>
                <p className="text-muted text-xs mt-1">
                  {new Date(project.updated_at * 1000).toLocaleString('fr-FR')} · {project.status} · {project.clips} clip(s)
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button type="button" className="btn-ghost text-xs" disabled={!!opening}
                  onClick={() => open(project.job_id)}>
                  {opening === project.job_id ? 'Ouverture…' : 'Ouvrir'}
                </button>
                {active && <button type="button" className="btn-ghost text-xs text-danger"
                  onClick={() => stop(project.job_id)}>Arrêter</button>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
