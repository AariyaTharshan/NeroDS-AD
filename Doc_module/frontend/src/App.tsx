import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import './App.css'

type Role = 'doctor' | 'patient'
type DoctorStage = 'login' | 'workspace'
type PatientStage = 'login' | 'history' | 'result'

interface DoctorResult {
  prediction_id: number
  patient_id: string
  dob: string
  diagnosis: string
  diagnostic_band: string
  certainty_label: string
  certainty_note: string
  patient_summary: string
  recommended_next_step: string
  created_at: string
  patient_scale: { headline: string; certainty: string; explanation: string }
  clinician_summary: string
  raw_cluster: number
  raw_cluster_label: string
  diagnosis_scores: Record<string, { value: number; label: string }>
  raw_cluster_scores: Record<string, { value: number; label: string }>
  mri_image: string
  pet_image: string
  mri_gradcam_image: string
  pet_gradcam_image: string
}

interface PatientResult {
  prediction_id: number
  patient_id: string
  dob: string
  diagnosis: string
  diagnostic_band: string
  certainty_label: string
  certainty_note: string
  patient_summary: string
  recommended_next_step: string
  created_at: string
  patient_scale: { headline: string; certainty: string; explanation: string }
  mri_gradcam_image: string
  pet_gradcam_image: string
}

interface PatientHistoryItem {
  prediction_id: number
  diagnosis: string
  diagnostic_band: string
  certainty_label: string
  created_at: string
}

const API_BASE = 'http://localhost:5050'
const STORAGE_KEY = 'doc_module_portal_session'

function labelForDiagnosis(diagnosis: string) {
  if (diagnosis === 'CN') return 'Cognitively Normal'
  if (diagnosis === 'MCI') return 'Mild Cognitive Impairment'
  return "Alzheimer's Disease"
}

function scoreTone(diagnosis: string) {
  if (diagnosis === 'AD') return 'ad'
  if (diagnosis === 'MCI') return 'mci'
  return 'cn'
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function App() {
  const [activeRole, setActiveRole] = useState<Role>('doctor')
  const [doctorStage, setDoctorStage] = useState<DoctorStage>('login')
  const [patientStage, setPatientStage] = useState<PatientStage>('login')
  const [doctorSession, setDoctorSession] = useState('')
  const [doctorLoading, setDoctorLoading] = useState(false)
  const [patientLoading, setPatientLoading] = useState(false)
  const [downloadLoading, setDownloadLoading] = useState(false)
  const [doctorError, setDoctorError] = useState('')
  const [patientError, setPatientError] = useState('')
  const [doctorResult, setDoctorResult] = useState<DoctorResult | null>(null)
  const [patientResult, setPatientResult] = useState<PatientResult | null>(null)
  const [patientHistory, setPatientHistory] = useState<PatientHistoryItem[]>([])
  const [doctorLogin, setDoctorLogin] = useState({ username: '', password: '' })
  const [doctorForm, setDoctorForm] = useState({
    patientId: '',
    dob: '',
    notes: '',
    mriImage: null as File | null,
    petImage: null as File | null,
  })
  const [patientForm, setPatientForm] = useState({ patientId: '', dob: '' })

  const themeTone = useMemo(
    () => scoreTone(doctorResult?.diagnosis ?? patientResult?.diagnosis ?? 'CN'),
    [doctorResult?.diagnosis, patientResult?.diagnosis],
  )

  const showDoctorWorkspace = activeRole === 'doctor' && doctorStage === 'workspace'
  const showPatientWorkspace = activeRole === 'patient' && patientStage !== 'login'
  const showHeroBanner = showDoctorWorkspace || showPatientWorkspace

  const doctorMetrics = useMemo(() => {
    if (!doctorResult) {
      return { mriFiles: '-', petFiles: '-', confidence: '-' }
    }
    const highest = Object.values(doctorResult.diagnosis_scores)[0]?.value ?? 0
    return {
      mriFiles: doctorForm.mriImage ? '1' : '-',
      petFiles: doctorForm.petImage ? '1' : '-',
      confidence: `${highest.toFixed(1)}%`,
    }
  }, [doctorForm.mriImage, doctorForm.petImage, doctorResult])

  const resetDoctorWorkspace = () => {
    setDoctorResult(null)
    setDoctorError('')
    setDoctorForm({ patientId: '', dob: '', notes: '', mriImage: null, petImage: null })
  }

  const logoutDoctor = () => {
    setDoctorStage('login')
    setDoctorSession('')
    setDoctorLogin({ username: '', password: '' })
    resetDoctorWorkspace()
  }

  const logoutPatient = () => {
    setPatientStage('login')
    setPatientError('')
    setPatientResult(null)
    setPatientHistory([])
    setPatientForm({ patientId: '', dob: '' })
  }

  const submitDoctorLogin = async (event: FormEvent) => {
    event.preventDefault()
    setDoctorLoading(true)
    setDoctorError('')
    try {
      const response = await fetch(`${API_BASE}/api/doctor/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(doctorLogin),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Login failed')
      setDoctorSession(data.username)
      setDoctorStage('workspace')
    } catch (error) {
      setDoctorError(error instanceof Error ? error.message : 'Login failed')
    } finally {
      setDoctorLoading(false)
    }
  }

  const handleDoctorFile = (field: 'mriImage' | 'petImage') => (event: ChangeEvent<HTMLInputElement>) => {
    setDoctorForm((state) => ({ ...state, [field]: event.target.files?.[0] ?? null }))
  }

  const submitDoctorPrediction = async (event: FormEvent) => {
    event.preventDefault()
    setDoctorLoading(true)
    setDoctorError('')
    setDoctorResult(null)

    try {
      const body = new FormData()
      body.append('patient_id', doctorForm.patientId)
      body.append('dob', doctorForm.dob)
      if (doctorForm.mriImage) body.append('mri_image', doctorForm.mriImage)
      if (doctorForm.petImage) body.append('pet_image', doctorForm.petImage)

      const response = await fetch(`${API_BASE}/api/doctor/predict`, { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Prediction failed')
      setDoctorResult(data)
    } catch (error) {
      setDoctorError(error instanceof Error ? error.message : 'Prediction failed')
    } finally {
      setDoctorLoading(false)
    }
  }

  const exportDoctorReport = async () => {
    if (!doctorResult) return
    setDownloadLoading(true)
    setDoctorError('')
    try {
      const response = await fetch(`${API_BASE}/api/doctor/report/${doctorResult.prediction_id}/download`)
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || 'Export failed')
      }
      const blob = await response.blob()
      downloadBlob(blob, `doctor_report_${doctorResult.patient_id}_${doctorResult.prediction_id}.pdf`)
    } catch (error) {
      setDoctorError(error instanceof Error ? error.message : 'Export failed')
    } finally {
      setDownloadLoading(false)
    }
  }

  const submitPatientLogin = async (event: FormEvent) => {
    event.preventDefault()
    setPatientLoading(true)
    setPatientError('')
    setPatientResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/patient/history`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: patientForm.patientId, dob: patientForm.dob }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Login failed')
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error('No reports found for that patient ID and date of birth')
      }
      setPatientHistory(data)
      setPatientStage('history')
    } catch (error) {
      setPatientError(error instanceof Error ? error.message : 'Login failed')
    } finally {
      setPatientLoading(false)
    }
  }

  const openPatientReport = async (predictionId: number) => {
    setPatientLoading(true)
    setPatientError('')
    try {
      const response = await fetch(`${API_BASE}/api/patient/report/${predictionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: patientForm.patientId, dob: patientForm.dob }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Could not open report')
      setPatientResult(data)
      setPatientStage('result')
    } catch (error) {
      setPatientError(error instanceof Error ? error.message : 'Could not open report')
    } finally {
      setPatientLoading(false)
    }
  }

  const downloadPatientReport = async () => {
    if (!patientResult) return
    setDownloadLoading(true)
    setPatientError('')
    try {
      const response = await fetch(`${API_BASE}/api/patient/report/${patientResult.prediction_id}/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: patientForm.patientId, dob: patientForm.dob }),
      })
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || 'Download failed')
      }
      const blob = await response.blob()
      downloadBlob(blob, `patient_report_${patientResult.patient_id}_${patientResult.prediction_id}.pdf`)
    } catch (error) {
      setPatientError(error instanceof Error ? error.message : 'Download failed')
    } finally {
      setDownloadLoading(false)
    }
  }

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (!saved) return
      const session = JSON.parse(saved) as {
        activeRole?: Role
        doctorSession?: string
        patientForm?: { patientId: string; dob: string }
      }

      if (session.activeRole === 'doctor' && session.doctorSession) {
        setActiveRole('doctor')
        setDoctorSession(session.doctorSession)
        setDoctorStage('workspace')
      }

      if (
        session.activeRole === 'patient' &&
        session.patientForm?.patientId &&
        session.patientForm?.dob
      ) {
        setActiveRole('patient')
        setPatientForm(session.patientForm)
        setPatientStage('history')
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY)
    }
  }, [])

  useEffect(() => {
    if (doctorStage === 'workspace' && doctorSession) {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          activeRole: 'doctor',
          doctorSession,
        }),
      )
      return
    }

    if (patientStage !== 'login' && patientForm.patientId && patientForm.dob) {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          activeRole: 'patient',
          patientForm,
        }),
      )
      return
    }

    localStorage.removeItem(STORAGE_KEY)
  }, [doctorSession, doctorStage, patientForm, patientStage])

  useEffect(() => {
    const loadPatientHistory = async () => {
      if (activeRole !== 'patient' || patientStage === 'login' || !patientForm.patientId || !patientForm.dob) {
        return
      }

      try {
        const response = await fetch(`${API_BASE}/api/patient/history`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_id: patientForm.patientId, dob: patientForm.dob }),
        })
        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.error || 'Could not restore patient session')
        }
        setPatientHistory(Array.isArray(data) ? data : [])
      } catch (error) {
        setPatientError(error instanceof Error ? error.message : 'Could not restore patient session')
      }
    }

    void loadPatientHistory()
  }, [activeRole, patientForm, patientStage])

  return (
    <div className={`app-shell tone-${themeTone}`}>
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-badge">N</div>
          <div>
            <strong>NEURO DS</strong>
            <span>Multimodal Alzheimer Console</span>
          </div>
        </div>
        <div className="topbar-actions">
          {activeRole === 'doctor' ? (
            <button className="topbar-chip active">Doctor Mode</button>
          ) : (
            <button className="topbar-chip active">Patient Mode</button>
          )}
          {activeRole === 'doctor' && doctorStage === 'workspace' ? <button className="topbar-chip" onClick={logoutDoctor}>Logout ({doctorSession})</button> : null}
          {activeRole === 'patient' && patientStage !== 'login' ? <button className="topbar-chip" onClick={logoutPatient}>Logout ({patientForm.patientId})</button> : null}
        </div>
      </header>

      <main className="page">
        {showHeroBanner ? (
          <section className="hero-banner">
            <div className="hero-copy">
            <span className="section-tag">Alzheimer Diagnostic Console</span>
            <h1>
              Multimodal Deep Learning for Alzheimer&apos;s Disease Detection
              <br />
              Using MRI and PET Imaging
            </h1>
            <p>
              Upload paired MRI and PET studies, run model inference, review Grad-CAM evidence, and export a clinical report from a structured portal.
            </p>
            </div>
            <div className="metric-strip">
              <div className="metric-card"><span>MRI Files</span><strong>{doctorMetrics.mriFiles}</strong></div>
              <div className="metric-card"><span>PET Files</span><strong>{doctorMetrics.petFiles}</strong></div>
              <div className="metric-card"><span>Confidence</span><strong>{doctorMetrics.confidence}</strong></div>
            </div>
          </section>
        ) : null}

        {activeRole === 'doctor' ? (
          doctorStage === 'login' ? (
            <section className="login-panel">
              <div className="console-card login-card">
                <div className="card-kicker">Doctor Access</div>
                <h2>Login Page</h2>
                <p>Use doctor username and password to access the imaging workflow console.</p>
                <form className="stack-form" onSubmit={submitDoctorLogin}>
                  <label>
                    Username
                    <input value={doctorLogin.username} onChange={(event) => setDoctorLogin((state) => ({ ...state, username: event.target.value }))} placeholder="doctor" required />
                  </label>
                  <label>
                    Password
                    <input type="password" value={doctorLogin.password} onChange={(event) => setDoctorLogin((state) => ({ ...state, password: event.target.value }))} placeholder="doctor123" required />
                  </label>
                  <button className="primary-action" type="submit" disabled={doctorLoading}>
                    {doctorLoading ? 'Signing in...' : 'Open Doctor Console'}
                  </button>
                </form>
                <button className="text-switch" type="button" onClick={() => setActiveRole('patient')}>
                  Switch to patient login
                </button>
                <div className="helper-note">Default login is <strong>doctor</strong> / <strong>doctor123</strong>. You can override it in backend environment variables.</div>
                {doctorError ? <div className="error-box">{doctorError}</div> : null}
              </div>
            </section>
          ) : (
            <section className="workspace-grid">
              <form className="console-card input-console" onSubmit={submitDoctorPrediction}>
                <div className="card-headline">
                  <div>
                    <span className="card-kicker">Input Workflow</span>
                    <h2>Patient details and studies</h2>
                  </div>
                  <span className="card-code">MOD-01</span>
                </div>

                <div className="inline-grid two-up">
                  <label>
                    Patient ID
                    <input value={doctorForm.patientId} onChange={(event) => setDoctorForm((state) => ({ ...state, patientId: event.target.value }))} placeholder="AD-1025" required />
                  </label>
                  <label>
                    Date of Birth
                    <input type="date" value={doctorForm.dob} onChange={(event) => setDoctorForm((state) => ({ ...state, dob: event.target.value }))} required />
                  </label>
                </div>

                <label>
                  Clinical Notes
                  <input value={doctorForm.notes} onChange={(event) => setDoctorForm((state) => ({ ...state, notes: event.target.value }))} placeholder="Optional context" />
                </label>

                <div className="study-grid">
                  <div className="study-card">
                    <div className="study-head">
                      <strong>MRI Study</strong>
                      <span>DICOM</span>
                    </div>
                    <input type="file" accept=".dcm,image/png,image/jpeg" onChange={handleDoctorFile('mriImage')} required />
                    <small>{doctorForm.mriImage ? doctorForm.mriImage.name : 'No file selected'}</small>
                  </div>
                  <div className="study-card">
                    <div className="study-head">
                      <strong>PET Study</strong>
                      <span>DICOM</span>
                    </div>
                    <input type="file" accept=".dcm,image/png,image/jpeg" onChange={handleDoctorFile('petImage')} required />
                    <small>{doctorForm.petImage ? doctorForm.petImage.name : 'No file selected'}</small>
                  </div>
                </div>

                <div className="action-row">
                  <button className="primary-action" type="submit" disabled={doctorLoading}>
                    {doctorLoading ? 'Running Prediction...' : 'Run Prediction'}
                  </button>
                  <button className="secondary-action" type="button" onClick={exportDoctorReport} disabled={!doctorResult || downloadLoading}>
                    {downloadLoading ? 'Exporting PDF...' : 'Export PDF Report'}
                  </button>
                  <button className="secondary-action" type="button" onClick={resetDoctorWorkspace}>Reset</button>
                </div>
                {doctorError ? <div className="error-box">{doctorError}</div> : null}
              </form>

              <section className="console-card output-console">
                <div className="card-headline">
                  <div>
                    <span className="card-kicker">Diagnostic Output</span>
                    <h2>{doctorResult ? labelForDiagnosis(doctorResult.diagnosis) : 'Awaiting inference request'}</h2>
                  </div>
                  <span className="card-code">INFER-01</span>
                </div>

                {!doctorResult ? (
                  <div className="empty-state">
                    <div className="pulse-ring" />
                    <p>Upload paired MRI and PET studies to generate the diagnostic output.</p>
                  </div>
                ) : (
                  <div className="output-stack">
                    <div className={`result-badge ${doctorResult.diagnosis.toLowerCase()}`}>{doctorResult.diagnosis}</div>
                    <div className="summary-box">
                      <strong>{doctorResult.diagnostic_band}</strong>
                      <p>{doctorResult.clinician_summary}</p>
                    </div>
                    <div className="mini-stats">
                      <div><span>Patient</span><strong>{doctorResult.patient_id}</strong></div>
                      <div><span>Certainty</span><strong>{doctorResult.certainty_label}</strong></div>
                      <div><span>Cluster</span><strong>{doctorResult.raw_cluster_label}</strong></div>
                    </div>
                    <div className="score-list">
                      {Object.entries(doctorResult.diagnosis_scores).map(([label, score]) => (
                        <div key={label} className="score-line">
                          <div className="score-line-head"><span>{label}</span><strong>{score.value.toFixed(1)}%</strong></div>
                          <div className="score-track"><div style={{ width: `${score.value}%` }} /></div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>

              {doctorResult ? (
                <>
                  <section className="console-card wide-card">
                    <div className="card-headline">
                      <div>
                        <span className="card-kicker">Grad-CAM</span>
                        <h2>MRI and PET explainability</h2>
                      </div>
                      <span className="card-code">VIS-01</span>
                    </div>
                    <div className="image-grid">
                      <figure>
                        <img src={`data:image/png;base64,${doctorResult.mri_image}`} alt="MRI original" />
                        <figcaption>Original MRI</figcaption>
                      </figure>
                      <figure>
                        <img src={`data:image/png;base64,${doctorResult.mri_gradcam_image}`} alt="MRI Grad-CAM" />
                        <figcaption>MRI Grad-CAM</figcaption>
                      </figure>
                      <figure>
                        <img src={`data:image/png;base64,${doctorResult.pet_image}`} alt="PET original" />
                        <figcaption>Original PET</figcaption>
                      </figure>
                      <figure>
                        <img src={`data:image/png;base64,${doctorResult.pet_gradcam_image}`} alt="PET Grad-CAM" />
                        <figcaption>PET Grad-CAM</figcaption>
                      </figure>
                    </div>
                  </section>

                  <section className="console-card wide-card">
                    <div className="card-headline">
                      <div>
                        <span className="card-kicker">Clinical Guidance</span>
                        <h2>Recommended next step</h2>
                      </div>
                      <span className="card-code">CARE-01</span>
                    </div>
                    <div className="plain-grid">
                      <div className="summary-box">
                        <strong>Doctor note</strong>
                        <p>{doctorResult.recommended_next_step}</p>
                      </div>
                      <div className="summary-box">
                        <strong>Patient explanation</strong>
                        <p>{doctorResult.patient_summary}</p>
                      </div>
                    </div>
                  </section>
                </>
              ) : null}
            </section>
          )
        ) : (
          patientStage === 'login' ? (
            <section className="login-panel">
              <div className="console-card login-card">
                <div className="card-kicker">Patient Access</div>
                <h2>Login Page</h2>
                <p>Use patient ID and date of birth to open report history and download the patient-facing result.</p>
                <form className="stack-form" onSubmit={submitPatientLogin}>
                  <label>
                    Patient ID
                    <input value={patientForm.patientId} onChange={(event) => setPatientForm((state) => ({ ...state, patientId: event.target.value }))} placeholder="AD-1025" required />
                  </label>
                  <label>
                    Date of Birth
                    <input type="date" value={patientForm.dob} onChange={(event) => setPatientForm((state) => ({ ...state, dob: event.target.value }))} required />
                  </label>
                  <button className="primary-action" type="submit" disabled={patientLoading}>
                    {patientLoading ? 'Checking records...' : 'Open Patient History'}
                  </button>
                </form>
                <button className="text-switch" type="button" onClick={() => setActiveRole('doctor')}>
                  Switch to doctor login
                </button>
                {patientError ? <div className="error-box">{patientError}</div> : null}
              </div>
            </section>
          ) : (
            <section className="workspace-grid patient-workspace">
              <section className="console-card input-console">
                <div className="card-headline">
                  <div>
                    <span className="card-kicker">History</span>
                    <h2>Available reports</h2>
                  </div>
                  <span className="card-code">PAT-01</span>
                </div>
                <div className="history-list">
                  {patientHistory.map((item) => (
                    <button key={item.prediction_id} className="history-item" onClick={() => openPatientReport(item.prediction_id)}>
                      <div>
                        <strong>{labelForDiagnosis(item.diagnosis)}</strong>
                        <span>{item.diagnostic_band}</span>
                      </div>
                      <small>{item.created_at}</small>
                    </button>
                  ))}
                </div>
                {patientError ? <div className="error-box">{patientError}</div> : null}
              </section>

              <section className="console-card output-console">
                <div className="card-headline">
                  <div>
                    <span className="card-kicker">Result</span>
                    <h2>{patientResult ? labelForDiagnosis(patientResult.diagnosis) : 'Choose a report from history'}</h2>
                  </div>
                  <span className="card-code">PAT-02</span>
                </div>
                {!patientResult ? (
                  <div className="empty-state">
                    <div className="pulse-ring" />
                    <p>Select a report to see the plain-language result.</p>
                  </div>
                ) : (
                  <div className="output-stack">
                    <div className={`result-badge ${patientResult.diagnosis.toLowerCase()}`}>{patientResult.diagnosis}</div>
                    <div className="summary-box">
                      <strong>{patientResult.diagnostic_band}</strong>
                      <p>{patientResult.patient_summary}</p>
                    </div>
                    <div className="mini-stats">
                      <div><span>Result meaning</span><strong>{patientResult.diagnostic_band}</strong></div>
                      <div><span>How certain</span><strong>{patientResult.certainty_label}</strong></div>
                      <div><span>Next step</span><strong>{patientResult.recommended_next_step}</strong></div>
                    </div>
                    <div className="action-row single-line">
                      <button className="secondary-action" type="button" onClick={downloadPatientReport} disabled={downloadLoading}>
                        {downloadLoading ? 'Preparing PDF...' : 'Download PDF Report'}
                      </button>
                    </div>
                  </div>
                )}
              </section>

              {patientResult ? (
                <section className="console-card wide-card">
                  <div className="card-headline">
                    <div>
                      <span className="card-kicker">Explainability</span>
                      <h2>Grad-CAM review</h2>
                    </div>
                    <span className="card-code">PAT-03</span>
                  </div>
                  <div className="image-grid two-up-images">
                    <figure>
                      <img src={`data:image/png;base64,${patientResult.mri_gradcam_image}`} alt="MRI Grad-CAM" />
                      <figcaption>MRI attention map</figcaption>
                    </figure>
                    <figure>
                      <img src={`data:image/png;base64,${patientResult.pet_gradcam_image}`} alt="PET Grad-CAM" />
                      <figcaption>PET attention map</figcaption>
                    </figure>
                  </div>
                </section>
              ) : null}
            </section>
          )
        )}
      </main>
    </div>
  )
}

export default App
