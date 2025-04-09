from fpdf import FPDF
import json

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Cybersecurity Statistics Report', 0, 1, 'C')
        self.ln(20)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, data):
        self.set_font('Arial', '', 12)
        for key, value in data.items():
            self.cell(0, 10, f"{key}: {value}", 0, 1)
        self.ln(10)

# Create PDF
pdf = PDF()
pdf.add_page()

# Add sections
sections = {
    "Key Metrics": {
        "Incident Response Times": "Average: 5.2 hours, Median: 4.8 hours, 95th percentile: 12.3 hours",
        "Security Events": "Total: 1,250, Critical: 120 (9.6%), High: 250 (20%), Medium: 500 (40%), Low: 380 (30.4%)",
        "Patch Management": "Average deployment: 3.5 days, Critical patches: 100% within 24h, High severity: 95% within 72h",
        "User Authentication": "Average attempts: 2.1, Failed attempts: 0.8%, MFA usage: 85%"
    }
}

for section, data in sections.items():
    pdf.chapter_title(section)
    pdf.chapter_body(data)

# Save PDF
pdf.output('sample_data.pdf')
