-- ============================================================
--  ANPR Intelligent Vehicle Monitoring & Theft Detection System
--  Database Schema  |  PostgreSQL 14+
-- ============================================================

-- UUID support
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- ============================================================
--  ENUM TYPES
-- ============================================================

CREATE TYPE vehicle_status   AS ENUM ('Clear', 'Stolen/Missing', 'Recovered');
CREATE TYPE complaint_status AS ENUM ('Active', 'Resolved', 'Closed');
CREATE TYPE alert_type_enum  AS ENUM ('STOLEN', 'EXPIRED_REGISTRATION', 'EXPIRED_INSURANCE', 'WATCHLIST', 'UNKNOWN_PLATE');


-- ============================================================
--  LOOKUP TABLE: rto_state_codes
--  Maps 2-letter state prefix to full state/UT name.
--  Used for display, validation, and JOIN with vehicles.registered_rto_state.
-- ============================================================

CREATE TABLE rto_state_codes (
    state_code  CHAR(2)     NOT NULL,
    state_name  VARCHAR(60) NOT NULL,
    territory_type VARCHAR(10) NOT NULL DEFAULT 'State',  -- 'State' | 'UT' | 'Special'
    CONSTRAINT pk_rto_state_codes PRIMARY KEY (state_code)
);

INSERT INTO rto_state_codes (state_code, state_name, territory_type) VALUES
-- States
('AP', 'Andhra Pradesh',                         'State'),
('AR', 'Arunachal Pradesh',                      'State'),
('AS', 'Assam',                                  'State'),
('BR', 'Bihar',                                  'State'),
('CG', 'Chhattisgarh',                           'State'),
('GA', 'Goa',                                    'State'),
('GJ', 'Gujarat',                                'State'),
('HR', 'Haryana',                                'State'),
('HP', 'Himachal Pradesh',                       'State'),
('JH', 'Jharkhand',                              'State'),
('JK', 'Jammu and Kashmir',                      'State'),
('KA', 'Karnataka',                              'State'),
('KL', 'Kerala',                                 'State'),
('LA', 'Ladakh',                                 'State'),
('MP', 'Madhya Pradesh',                         'State'),
('MH', 'Maharashtra',                            'State'),
('MN', 'Manipur',                                'State'),
('ML', 'Meghalaya',                              'State'),
('MZ', 'Mizoram',                                'State'),
('NL', 'Nagaland',                               'State'),
('OD', 'Odisha',                                 'State'),
('PB', 'Punjab',                                 'State'),
('RJ', 'Rajasthan',                              'State'),
('SK', 'Sikkim',                                 'State'),
('TN', 'Tamil Nadu',                             'State'),
('TG', 'Telangana',                              'State'),
('TR', 'Tripura',                                'State'),
('UK', 'Uttarakhand',                            'State'),
('UP', 'Uttar Pradesh',                          'State'),
('WB', 'West Bengal',                            'State'),
-- Union Territories
('CH', 'Chandigarh',                             'UT'),
('DD', 'Dadra and Nagar Haveli and Daman & Diu', 'UT'),
('DL', 'Delhi',                                  'UT'),
('LD', 'Lakshadweep',                            'UT'),
('PY', 'Puducherry',                             'UT'),
-- Special / pan-India
('BH', 'Bharat Series (Pan-India)',              'Special');

-- FK from vehicles to the lookup table (state prefix only, first 2 chars of rto_code)
ALTER TABLE vehicles
    ADD COLUMN rto_state_prefix CHAR(2)
        GENERATED ALWAYS AS (UPPER(SUBSTRING(registered_rto_code, 1, 2))) STORED,
    ADD CONSTRAINT fk_vehicles_rto_state
        FOREIGN KEY (rto_state_prefix)
        REFERENCES rto_state_codes(state_code);


-- ============================================================
--  USEFUL QUERY: resolve plate prefix → state name
-- ============================================================
-- SELECT v.plate_number,
--        v.registered_rto_code,
--        r.state_name,
--        r.territory_type
-- FROM   vehicles v
-- JOIN   rto_state_codes r ON r.state_code = v.rto_state_prefix;


-- ============================================================
--  TABLE: vehicles
--  Core registry – one row per unique license plate
-- ============================================================

CREATE TABLE vehicles (

    -- Primary identifier
    plate_number            VARCHAR(20)     NOT NULL,

    -- Vehicle details
    vehicle_make            VARCHAR(50)     NOT NULL,
    vehicle_model           VARCHAR(60)     NOT NULL,
    vehicle_year            SMALLINT        CHECK (vehicle_year BETWEEN 1900 AND 2100),
    vehicle_color           VARCHAR(40)     NOT NULL,   -- used for spoof/clone detection
    vehicle_type            VARCHAR(30)     NOT NULL DEFAULT 'Car',   -- Car/Bike/Truck/Bus/Auto

    -- Owner information
    owner_name              VARCHAR(100)    NOT NULL,
    owner_phone             VARCHAR(15),
    owner_email             VARCHAR(100),
    owner_address           TEXT,
    owner_aadhaar_last4     CHAR(4),        -- last 4 digits only for cross-verification

    -- RTO registration
    registered_rto_state    VARCHAR(50)     NOT NULL,
    registered_rto_code     VARCHAR(10)     NOT NULL,   -- e.g. 'TS32', 'AP28', 'MH01'
    chassis_number          VARCHAR(50)     UNIQUE,
    engine_number           VARCHAR(50),
    registration_date       DATE,
    registration_expiry     DATE,
    insurance_expiry        DATE,

    -- Status & theft tracking
    status                  vehicle_status  NOT NULL DEFAULT 'Clear',
    police_complaint_id     VARCHAR(60),                -- FK to active theft_reports.complaint_id
    missing_date            TIMESTAMPTZ,
    recovery_date           TIMESTAMPTZ,

    -- Audit
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_vehicles PRIMARY KEY (plate_number),

    -- Recovery must be after the missing date
    CONSTRAINT chk_recovery_after_missing CHECK (
        recovery_date IS NULL
        OR missing_date IS NULL
        OR recovery_date >= missing_date
    ),

    -- missing_date must be set when status is Stolen/Missing
    CONSTRAINT chk_stolen_has_missing_date CHECK (
        status != 'Stolen/Missing' OR missing_date IS NOT NULL
    )
);


-- ============================================================
--  TABLE: theft_reports
--  Full audit trail – every filed complaint (historic + active)
-- ============================================================

CREATE TABLE theft_reports (

    report_id               SERIAL          PRIMARY KEY,
    plate_number            VARCHAR(20)     NOT NULL
                                REFERENCES vehicles(plate_number)
                                ON UPDATE CASCADE ON DELETE RESTRICT,
    complaint_id            VARCHAR(60)     NOT NULL UNIQUE,

    -- Filing details
    reported_by             VARCHAR(100),
    reporting_station       VARCHAR(120),   -- e.g. 'Tambaram Police Station'
    reporting_district      VARCHAR(60),
    theft_location          TEXT,
    theft_description       TEXT,

    -- Timestamps
    theft_datetime          TIMESTAMPTZ,
    filed_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolved_at             TIMESTAMPTZ,

    -- Investigation
    report_status           complaint_status NOT NULL DEFAULT 'Active',
    investigating_officer   VARCHAR(100),
    officer_badge_number    VARCHAR(30),
    resolution_notes        TEXT,

    CONSTRAINT chk_resolved_after_filed CHECK (
        resolved_at IS NULL OR resolved_at >= filed_at
    )
);


-- ============================================================
--  TABLE: anpr_scan_log
--  Every plate read by every camera (append-only event log)
-- ============================================================

CREATE TABLE anpr_scan_log (

    scan_id                 UUID            NOT NULL DEFAULT gen_random_uuid(),
    plate_number            VARCHAR(20),    -- NULL when OCR failed / unreadable
    raw_ocr_text            VARCHAR(40),    -- raw TrOCR output before regex cleaning
    confidence_score        NUMERIC(5, 4)   CHECK (confidence_score BETWEEN 0 AND 1),

    -- Camera / location
    camera_id               VARCHAR(50)     NOT NULL,
    camera_location         VARCHAR(180),
    scan_timestamp          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Alert
    alert_triggered         BOOLEAN         NOT NULL DEFAULT FALSE,
    alert_type              alert_type_enum,
    alert_acknowledged      BOOLEAN         NOT NULL DEFAULT FALSE,
    acknowledged_by         VARCHAR(100),
    acknowledged_at         TIMESTAMPTZ,

    -- Asset paths (served via FastAPI /static)
    image_path              VARCHAR(255),
    vehicle_crop_path       VARCHAR(255),
    plate_crop_path         VARCHAR(255),

    CONSTRAINT pk_anpr_scan_log PRIMARY KEY (scan_id),

    CONSTRAINT chk_ack_requires_alert CHECK (
        alert_acknowledged = FALSE OR alert_triggered = TRUE
    )
);


-- ============================================================
--  INDEXES
-- ============================================================

-- Vehicles – frequent filter columns
CREATE INDEX idx_vehicles_status        ON vehicles(status);
CREATE INDEX idx_vehicles_rto_code      ON vehicles(registered_rto_code);
CREATE INDEX idx_vehicles_owner_name    ON vehicles(owner_name);

-- Theft reports
CREATE INDEX idx_theft_plate            ON theft_reports(plate_number);
CREATE INDEX idx_theft_complaint        ON theft_reports(complaint_id);
CREATE INDEX idx_theft_status           ON theft_reports(report_status);

-- Scan log – high-volume, time-ordered
CREATE INDEX idx_scan_plate             ON anpr_scan_log(plate_number);
CREATE INDEX idx_scan_timestamp         ON anpr_scan_log(scan_timestamp DESC);
CREATE INDEX idx_scan_alerts            ON anpr_scan_log(alert_triggered, scan_timestamp DESC)
                                        WHERE alert_triggered = TRUE;
CREATE INDEX idx_scan_camera            ON anpr_scan_log(camera_id, scan_timestamp DESC);


-- ============================================================
--  TRIGGER: auto-set updated_at on vehicles
-- ============================================================

CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trig_vehicles_updated_at
    BEFORE UPDATE ON vehicles
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();


-- ============================================================
--  FUNCTION: file_police_complaint
--
--  Sets vehicle status → 'Stolen/Missing', stamps missing_date,
--  and inserts a record into theft_reports.
--
--  Returns: (success BOOLEAN, message TEXT)
--
--  Usage:
--    SELECT * FROM file_police_complaint(
--        'TS32T2514', 'FIR-TS-2026-00001',
--        'Owner Name', 'Local Police Station',
--        'Parking lot near mall', NULL
--    );
-- ============================================================

CREATE OR REPLACE FUNCTION file_police_complaint(
    p_plate_number        VARCHAR(20),
    p_complaint_id        VARCHAR(60),
    p_reported_by         VARCHAR(100)  DEFAULT NULL,
    p_reporting_station   VARCHAR(120)  DEFAULT NULL,
    p_theft_location      TEXT          DEFAULT NULL,
    p_theft_datetime      TIMESTAMPTZ   DEFAULT NULL
)
RETURNS TABLE(success BOOLEAN, message TEXT)
LANGUAGE plpgsql AS $$
DECLARE
    v_status vehicle_status;
BEGIN
    -- 1. Vehicle must exist
    SELECT status INTO v_status
    FROM   vehicles
    WHERE  plate_number = p_plate_number;

    IF NOT FOUND THEN
        RETURN QUERY
            SELECT FALSE,
                   format('Vehicle %s not found in registry.', p_plate_number);
        RETURN;
    END IF;

    -- 2. Complaint ID must be unique
    IF EXISTS (SELECT 1 FROM theft_reports WHERE complaint_id = p_complaint_id) THEN
        RETURN QUERY
            SELECT FALSE,
                   format('Complaint ID %s already exists.', p_complaint_id);
        RETURN;
    END IF;

    -- 3. Reject if already Stolen/Missing
    IF v_status = 'Stolen/Missing' THEN
        RETURN QUERY
            SELECT FALSE,
                   format('Vehicle %s is already marked Stolen/Missing.', p_plate_number);
        RETURN;
    END IF;

    -- 4. Update vehicle record
    UPDATE vehicles
    SET    status              = 'Stolen/Missing',
           police_complaint_id = p_complaint_id,
           missing_date        = NOW(),
           recovery_date       = NULL
    WHERE  plate_number = p_plate_number;

    -- 5. Insert theft report
    INSERT INTO theft_reports (
        plate_number, complaint_id,
        reported_by, reporting_station,
        theft_location, theft_datetime,
        filed_at, report_status
    ) VALUES (
        p_plate_number, p_complaint_id,
        p_reported_by, p_reporting_station,
        p_theft_location, COALESCE(p_theft_datetime, NOW()),
        NOW(), 'Active'
    );

    RETURN QUERY
        SELECT TRUE,
               format(
                   'Complaint %s filed. Vehicle %s is now Stolen/Missing.',
                   p_complaint_id, p_plate_number
               );
END;
$$;


-- ============================================================
--  FUNCTION: mark_vehicle_recovered
--
--  Sets vehicle status → 'Recovered', stamps recovery_date,
--  and closes the linked active theft_report.
--
--  Returns: (success BOOLEAN, message TEXT)
--
--  Usage:
--    SELECT * FROM mark_vehicle_recovered(
--        'TN22CK1193',
--        'Recovered during patrol near Saidapet',
--        'SI Murugesan P.'
--    );
-- ============================================================

CREATE OR REPLACE FUNCTION mark_vehicle_recovered(
    p_plate_number      VARCHAR(20),
    p_resolution_notes  TEXT         DEFAULT NULL,
    p_officer           VARCHAR(100) DEFAULT NULL
)
RETURNS TABLE(success BOOLEAN, message TEXT)
LANGUAGE plpgsql AS $$
DECLARE
    v_status       vehicle_status;
    v_complaint_id VARCHAR(60);
BEGIN
    SELECT status, police_complaint_id
    INTO   v_status, v_complaint_id
    FROM   vehicles
    WHERE  plate_number = p_plate_number;

    IF NOT FOUND THEN
        RETURN QUERY
            SELECT FALSE,
                   format('Vehicle %s not found in registry.', p_plate_number);
        RETURN;
    END IF;

    IF v_status != 'Stolen/Missing' THEN
        RETURN QUERY
            SELECT FALSE,
                   format(
                       'Vehicle %s is not currently Stolen/Missing (current status: %s).',
                       p_plate_number, v_status
                   );
        RETURN;
    END IF;

    -- Update vehicle
    UPDATE vehicles
    SET    status        = 'Recovered',
           recovery_date = NOW()
    WHERE  plate_number  = p_plate_number;

    -- Close the linked complaint
    UPDATE theft_reports
    SET    report_status        = 'Resolved',
           resolved_at          = NOW(),
           resolution_notes     = p_resolution_notes,
           investigating_officer = COALESCE(p_officer, investigating_officer)
    WHERE  complaint_id  = v_complaint_id
      AND  report_status = 'Active';

    RETURN QUERY
        SELECT TRUE,
               format(
                   'Vehicle %s marked Recovered. Complaint %s closed.',
                   p_plate_number, v_complaint_id
               );
END;
$$;


-- ============================================================
--  VIEW: stolen_vehicles
--  Fast lookup used by the ANPR alert engine at scan time
-- ============================================================

CREATE OR REPLACE VIEW stolen_vehicles AS
SELECT
    v.plate_number,
    v.vehicle_make,
    v.vehicle_model,
    v.vehicle_year,
    v.vehicle_color,
    v.vehicle_type,
    v.owner_name,
    v.owner_phone,
    v.registered_rto_state,
    v.police_complaint_id,
    v.missing_date,
    tr.reporting_station,
    tr.theft_location,
    tr.investigating_officer
FROM  vehicles v
LEFT  JOIN theft_reports tr
      ON  tr.complaint_id = v.police_complaint_id
WHERE v.status = 'Stolen/Missing';


-- ============================================================
--  VIEW: recent_alerts
--  Last 24 hours of triggered ANPR alerts with vehicle context
-- ============================================================

CREATE OR REPLACE VIEW recent_alerts AS
SELECT
    s.scan_id,
    s.plate_number,
    s.raw_ocr_text,
    s.confidence_score,
    s.camera_id,
    s.camera_location,
    s.scan_timestamp,
    s.alert_type,
    s.alert_acknowledged,
    v.vehicle_make,
    v.vehicle_model,
    v.vehicle_color,
    v.owner_name,
    v.police_complaint_id
FROM  anpr_scan_log s
LEFT  JOIN vehicles v ON v.plate_number = s.plate_number
WHERE s.alert_triggered = TRUE
  AND s.scan_timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY s.scan_timestamp DESC;


-- ============================================================
--  DUMMY DATA
-- ============================================================

-- ---- vehicles -----------------------------------------------
INSERT INTO vehicles (
    plate_number,   vehicle_make,   vehicle_model,  vehicle_year,
    vehicle_color,  vehicle_type,   owner_name,     owner_phone,
    owner_email,    owner_address,
    registered_rto_state, registered_rto_code,
    chassis_number, engine_number,
    registration_date, registration_expiry, insurance_expiry,
    status, police_complaint_id, missing_date, recovery_date
) VALUES

-- ── Clear vehicles ────────────────────────────────────────────
('TS32T2514',
 'Maruti Suzuki', 'Swift Dzire', 2021, 'Pearl White', 'Car',
 'Ravi Kumar Sharma', '9876543210', 'ravi.sharma@gmail.com',
 '12-4-567, Banjara Hills, Hyderabad - 500034',
 'Telangana', 'TS32',
 'MBLFA62ZLNM123456', 'K10CNEM123456',
 '2021-03-15', '2036-03-14', '2026-03-14',
 'Clear', NULL, NULL, NULL),

('AP28AL4708',
 'Hyundai', 'i20', 2019, 'Typhoon Silver', 'Car',
 'Priya Reddy', '9988776655', 'priya.reddy@outlook.com',
 '3-45 MG Road, Vijayawada - 520010',
 'Andhra Pradesh', 'AP28',
 'MALAM51BLKM987654', 'G4LAKM987654',
 '2019-07-20', '2034-07-19', '2025-07-19',
 'Clear', NULL, NULL, NULL),

('MH04CE8821',
 'Tata Motors', 'Nexon EV', 2022, 'Flame Red', 'Car',
 'Ankit Deshmukh', '8765432109', 'ankit.d@yahoo.com',
 'Flat 5B, Koregaon Park, Pune - 411001',
 'Maharashtra', 'MH04',
 'TATL90AB2NM445566', 'TPEMNM445566',
 '2022-11-01', '2037-10-31', '2027-10-31',
 'Clear', NULL, NULL, NULL),

('KA05MN3301',
 'Honda', 'City', 2020, 'Lunar Silver Metallic', 'Car',
 'Deepa Nair', '7654321098', 'deepa.nair@gmail.com',
 '18 Indiranagar 100ft Road, Bangalore - 560038',
 'Karnataka', 'KA05',
 'MAKESGH2XLM334455', 'L15B2334455',
 '2020-05-10', '2035-05-09', '2026-05-09',
 'Clear', NULL, NULL, NULL),

('DL09CAB5521',
 'Toyota', 'Innova Crysta', 2018, 'White Pearl', 'Car',
 'Suresh Gupta', '9871234560', 'suresh.g@rediffmail.com',
 'A-14, Vasant Vihar, New Delhi - 110057',
 'Delhi', 'DL09',
 'MHFYB8CDXJM556677', '2TRFEM556677',
 '2018-08-25', '2033-08-24', '2025-08-24',
 'Clear', NULL, NULL, NULL),

-- ── Stolen / Missing vehicles ─────────────────────────────────
('TN22CK1193',
 'Bajaj', 'Pulsar 150', 2020, 'Midnight Black', 'Bike',
 'Karthik Murugan', '9444123456', 'karthik.m@gmail.com',
 '22 Nehru Street, Tambaram, Chennai - 600045',
 'Tamil Nadu', 'TN22',
 'MD2DHDHZZRWJ12345', 'DHDHZRWJ12345',
 '2020-02-14', '2035-02-13', '2025-02-13',
 'Stolen/Missing', 'FIR-TN-2025-00891',
 '2025-11-03 08:30:00+05:30', NULL),

('RJ14GH7742',
 'Maruti Suzuki', 'Alto 800', 2017, 'Cerulean Blue', 'Car',
 'Ramesh Choudhary', '9414567890', 'ramesh.c@gmail.com',
 'B-12 Mansarovar Colony, Jaipur - 302020',
 'Rajasthan', 'RJ14',
 'MBLRC06AZDM778899', 'F8DZDM778899',
 '2017-06-30', '2032-06-29', '2024-06-29',
 'Stolen/Missing', 'FIR-RJ-2025-01234',
 '2025-12-15 21:00:00+05:30', NULL),

('UP16BT4490',
 'Honda', 'Activa 6G', 2021, 'Pearl Siren Blue', 'Bike',
 'Neha Singh', '8765098765', 'neha.singh@gmail.com',
 'H-56, Sector 20, Noida - 201301',
 'Uttar Pradesh', 'UP16',
 'ME4JF505XMT201234', 'JF50ET201234',
 '2021-09-12', '2036-09-11', '2026-09-11',
 'Stolen/Missing', 'FIR-UP-2026-00045',
 '2026-01-22 19:15:00+05:30', NULL),

-- ── Recovered vehicles ────────────────────────────────────────
('GJ01BC2288',
 'Ford', 'EcoSport', 2016, 'Canyon Ridge Brown', 'Car',
 'Nilesh Patel', '9825012345', 'nilesh.p@gmail.com',
 '7 Satellite Road, Ahmedabad - 380015',
 'Gujarat', 'GJ01',
 'MAJXXMKJ2GUK34567', 'M1DAGUK34567',
 '2016-04-18', '2031-04-17', '2024-04-17',
 'Recovered', 'FIR-GJ-2024-00563',
 '2024-08-10 06:00:00+05:30', '2024-09-05 14:30:00+05:30'),

('PB10HD9934',
 'Mahindra', 'Scorpio N', 2023, 'Napoli Black', 'Car',
 'Harpreet Sandhu', '9815887766', 'harpreet.s@gmail.com',
 'House 88, Model Town, Jalandhar - 144001',
 'Punjab', 'PB10',
 'MALCC3FXRPM567890', 'MHAWKPM567890',
 '2023-01-05', '2038-01-04', '2027-01-04',
 'Recovered', 'FIR-PB-2025-00312',
 '2025-07-04 22:00:00+05:30', '2025-08-19 11:00:00+05:30');


-- ---- theft_reports ------------------------------------------
INSERT INTO theft_reports (
    plate_number, complaint_id,
    reported_by, reporting_station, reporting_district,
    theft_location, theft_description,
    theft_datetime, filed_at,
    report_status, resolved_at, resolution_notes,
    investigating_officer, officer_badge_number
) VALUES

-- Stolen/Missing – Active complaints
('TN22CK1193', 'FIR-TN-2025-00891',
 'Karthik Murugan', 'Tambaram Police Station', 'Chennai',
 'Tambaram Railway Station west parking lot',
 'Bike parked overnight; found missing at 8:30 AM.',
 '2025-11-03 08:30:00+05:30', '2025-11-03 11:45:00+05:30',
 'Active', NULL, NULL,
 'SI Murugesan P.', 'TN-SI-4412'),

('RJ14GH7742', 'FIR-RJ-2025-01234',
 'Ramesh Choudhary', 'Mansarovar Police Station', 'Jaipur',
 'Outside owner''s residence, Mansarovar Colony',
 'Vehicle was parked in front of house; gone the next morning.',
 '2025-12-15 21:00:00+05:30', '2025-12-16 08:00:00+05:30',
 'Active', NULL, NULL,
 'ASI Bharat Lal', 'RJ-ASI-8821'),

('UP16BT4490', 'FIR-UP-2026-00045',
 'Neha Singh', 'Sector-20 Noida Police Station', 'Gautam Buddh Nagar',
 'Sector-18 market parking, Noida',
 'Scooter stolen from market parking while owner was shopping.',
 '2026-01-22 19:15:00+05:30', '2026-01-22 20:30:00+05:30',
 'Active', NULL, NULL,
 'SI Vikas Yadav', 'UP-SI-3391'),

-- Recovered – Resolved complaints
('GJ01BC2288', 'FIR-GJ-2024-00563',
 'Nilesh Patel', 'Navrangpura Police Station', 'Ahmedabad',
 'CG Road commercial area, Ahmedabad',
 'Vehicle stolen from commercial area. Owner noticed next morning.',
 '2024-08-10 06:00:00+05:30', '2024-08-10 09:30:00+05:30',
 'Resolved', '2024-09-05 14:30:00+05:30',
 'Vehicle recovered from scrapped-goods yard in Surat. VIN tampering detected. Suspect arrested.',
 'PI Rakesh Mehta', 'GJ-PI-1172'),

('PB10HD9934', 'FIR-PB-2025-00312',
 'Harpreet Sandhu', 'Jalandhar City Police Station', 'Jalandhar',
 'GT Road near Phillour; vehicle taken at gunpoint',
 'Armed robbery; two assailants on motorcycle forced victim to hand over keys.',
 '2025-07-04 22:00:00+05:30', '2025-07-05 01:00:00+05:30',
 'Resolved', '2025-08-19 11:00:00+05:30',
 'Vehicle recovered in Ludhiana. Two accused arrested by Punjab Police SIT. Chargesheet filed.',
 'DSP Gurinder Singh', 'PB-DSP-0055');


-- ---- anpr_scan_log ------------------------------------------
INSERT INTO anpr_scan_log (
    plate_number, raw_ocr_text, confidence_score,
    camera_id, camera_location, scan_timestamp,
    alert_triggered, alert_type,
    image_path, plate_crop_path
) VALUES

-- Normal scans (Clear vehicles)
('TS32T2514',  'TS32 T2514',   0.9920,
 'CAM-HYD-001', 'Hitech City Toll Gate, Hyderabad',
 NOW() - INTERVAL '2 hours',
 FALSE, NULL,
 '/static/scans/cam001_ts32t2514.jpg', '/static/crops/plate_ts32t2514.jpg'),

('AP28AL4708', 'AP28AL4708',   0.9918,
 'CAM-VJA-003', 'Vijayawada NH-16 Entry Checkpoint',
 NOW() - INTERVAL '4 hours',
 FALSE, NULL,
 '/static/scans/cam003_ap28al4708.jpg', '/static/crops/plate_ap28al4708.jpg'),

('MH04CE8821', 'MH04CE8821',   0.9750,
 'CAM-PUN-002', 'Pune–Mumbai Expressway KM 42',
 NOW() - INTERVAL '6 hours',
 FALSE, NULL,
 '/static/scans/cam002_mh04ce8821.jpg', '/static/crops/plate_mh04ce8821.jpg'),

('KA05MN3301', 'KA05 MN3301',  0.9610,
 'CAM-BLR-008', 'Silk Board Junction ANPR, Bangalore',
 NOW() - INTERVAL '1 hour  30 minutes',
 FALSE, NULL,
 '/static/scans/cam008_ka05mn3301.jpg', '/static/crops/plate_ka05mn3301.jpg'),

-- Stolen vehicle triggers
('TN22CK1193', 'TN22CK1193',   0.8750,
 'CAM-CHE-007', 'Tambaram Flyover Camera-7, Chennai',
 NOW() - INTERVAL '30 minutes',
 TRUE, 'STOLEN',
 '/static/scans/cam007_tn22ck1193.jpg', '/static/crops/plate_tn22ck1193.jpg'),

('RJ14GH7742', 'RJ14GH7742',   0.9100,
 'CAM-JAI-011', 'Jaipur Ring Road ANPR Point-11',
 NOW() - INTERVAL '55 minutes',
 TRUE, 'STOLEN',
 '/static/scans/cam011_rj14gh7742.jpg', '/static/crops/plate_rj14gh7742.jpg'),

('UP16BT4490', 'UP16 BT4490',  0.8840,
 'CAM-NOI-015', 'Noida Expressway Toll Booth-2',
 NOW() - INTERVAL '15 minutes',
 TRUE, 'STOLEN',
 '/static/scans/cam015_up16bt4490.jpg', '/static/crops/plate_up16bt4490.jpg'),

-- Expired insurance trigger
('DL09CAB5521','DL09CAB5521',  0.9450,
 'CAM-DEL-020', 'NH-48 Delhi–Gurgaon Checkpoint',
 NOW() - INTERVAL '3 hours',
 TRUE, 'EXPIRED_INSURANCE',
 '/static/scans/cam020_dl09cab5521.jpg', '/static/crops/plate_dl09cab5521.jpg'),

-- Low-confidence / unreadable plate (plate_number NULL)
(NULL,          'OB4X 7??9',   0.3200,
 'CAM-HYD-001', 'Hitech City Toll Gate, Hyderabad',
 NOW() - INTERVAL '45 minutes',
 FALSE, NULL,
 '/static/scans/cam001_unknown.jpg', '/static/crops/plate_unknown.jpg');


-- ============================================================
--  VERIFY: call the stored functions
-- ============================================================

-- 1. File a new complaint on a currently Clear vehicle
SELECT * FROM file_police_complaint(
    'TS32T2514',
    'FIR-TS-2026-00001',
    'Ravi Kumar Sharma',
    'Banjara Hills Police Station',
    'Hitech City IKEA parking, Level 2',
    NULL
);

-- 2. Attempt duplicate complaint (should fail)
SELECT * FROM file_police_complaint(
    'TS32T2514',
    'FIR-TS-2026-00001',  -- same ID
    'Someone Else', 'Another Station', NULL, NULL
);

-- 3. Recover a stolen vehicle
SELECT * FROM mark_vehicle_recovered(
    'TN22CK1193',
    'Recovered during routine patrol near Saidapet flyover. Suspect fled.',
    'SI Murugesan P.'
);

-- 4. Attempt to recover a non-stolen vehicle (should fail)
SELECT * FROM mark_vehicle_recovered('MH04CE8821');


-- ============================================================
--  USEFUL QUERIES
-- ============================================================

-- All currently stolen/missing vehicles (via view)
-- SELECT * FROM stolen_vehicles;

-- Recent 24-hour alerts with vehicle context (via view)
-- SELECT * FROM recent_alerts;

-- Full history of complaints for a plate
-- SELECT * FROM theft_reports WHERE plate_number = 'TN22CK1193' ORDER BY filed_at;

-- ANPR scan count per camera today
-- SELECT camera_id, camera_location, COUNT(*) AS scans,
--        SUM(alert_triggered::INT) AS alerts
-- FROM   anpr_scan_log
-- WHERE  scan_timestamp >= CURRENT_DATE
-- GROUP  BY camera_id, camera_location
-- ORDER  BY scans DESC;

-- Vehicles with expired insurance still being detected
-- SELECT DISTINCT s.plate_number, v.owner_name, v.insurance_expiry, v.owner_phone
-- FROM   anpr_scan_log s
-- JOIN   vehicles v ON v.plate_number = s.plate_number
-- WHERE  v.insurance_expiry < CURRENT_DATE
--   AND  s.scan_timestamp >= NOW() - INTERVAL '7 days';
