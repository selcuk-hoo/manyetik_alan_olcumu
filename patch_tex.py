import re

with open("makale_tr.tex", "r") as f:
    text = f.read()

# Replace Fig 1
text = re.sub(
    r'\\fbox\{\\parbox\{0\.75\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 1\]\}.*?\\vspace\{2cm\}\}\}',
    r'\\includegraphics[width=0.8\\textwidth]{fig1_falseedm_scan.png}',
    text,
    flags=re.DOTALL
)

# Replace Fig 2
text = re.sub(
    r'\\fbox\{\\parbox\{0\.75\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 2\]\}.*?\\vspace\{2cm\}\}\}',
    r'\\includegraphics[width=0.8\\textwidth]{fig2_orbit_gain.png}',
    text,
    flags=re.DOTALL
)

# Replace Fig 3
text = re.sub(
    r'\\fbox\{\\parbox\{0\.75\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 3\]\}.*?\\vspace\{2cm\}\}\}',
    r'\\includegraphics[width=1.0\\textwidth]{fig3_mode_patterns.png}',
    text,
    flags=re.DOTALL
)

# Replace Fig 4 & 5 (Delete the entire figure environments from Section 3)
text = re.sub(
    r'\\begin\{figure\}\[ht\]\n\\centering\n\\fbox\{\\parbox\{0\.75\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 4\]\}.*?\\end\{figure\}\n',
    '',
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'\\begin\{figure\}\[ht\]\n\\centering\n\\fbox\{\\parbox\{0\.75\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 5\]\}.*?\\end\{figure\}\n',
    '',
    text,
    flags=re.DOTALL
)

# Replace Fig 7 (Delete the entire figure environment from Section 1.3)
text = re.sub(
    r'\\begin\{figure\}\[ht\]\n\\centering\n\\fbox\{\\parbox\{0\.78\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 7\]\}.*?\\end\{figure\}\n',
    '',
    text,
    flags=re.DOTALL
)

# Replace Fig 6
text = re.sub(
    r'\\fbox\{\\parbox\{0\.75\\textwidth\}\{\\centering\\vspace\{2cm\}\n\\textbf\{\[ŞEKİL 6\]\}.*?\\vspace\{2cm\}\}\}',
    r'\\includegraphics[width=0.9\\textwidth]{fig6_combined_systematics.png}',
    text,
    flags=re.DOTALL
)

# Update Text References and Insert New Figures

# Section 4.1 (BPM gürültüsü)
text = text.replace(
    r"Analitik ölçekleme $\sigma(\Delta A/A)\propto\sigma/\|M_k\|$ simülasyonla" + "\n" + r"doğrulanmıştır.",
    r"Analitik ölçekleme simülasyonla doğrulanmıştır. Şekil~\ref{fig:noise_budget}'te gösterildiği gibi, artan $\sigma_b$ değerleri $k=1,2,3$ modlarının saptanmasında hata barlarını (istatistiksel varyansı) genişletirken ortalamada bir sapma yaratmamaktadır." + "\n\n" +
    r"\begin{figure}[ht]" + "\n" +
    r"\centering" + "\n" +
    r"\includegraphics[width=0.7\textwidth]{fig8_noise_model.png}" + "\n" +
    r"\caption{Saf BPM gürültüsünün $k=1,2,3$ modlarındaki göreli genlik tahminine etkisi. İdeal (hatasız) makine üzerine eklenen gürültü, tahminleyiciye sistematik sapma katmaz, ancak varyansı artırır.}" + "\n" +
    r"\label{fig:noise_budget}" + "\n" +
    r"\end{figure}"
)

# Section 4.2 (BPM ofseti)
text = text.replace(
    r"$\sigma_b=200\,\mu$m için $1{,}5\%$.",
    r"$\sigma_b=200\,\mu$m için $1{,}5\%$. Şekil~\ref{fig:offset_budget}'te gösterildiği gibi analitik öngörüler simülasyon verileriyle örtüşmektedir." + "\n\n" +
    r"\begin{figure}[ht]" + "\n" +
    r"\centering" + "\n" +
    r"\includegraphics[width=0.7\textwidth]{fig5_offset_whiteness.png}" + "\n" +
    r"\caption{BPM statik ofsetinin $k=1,2,3$ modlarındaki genlik hatasına etkisi. Analitik beklentiler simülasyon verileriyle örtüşmektedir.}" + "\n" +
    r"\label{fig:offset_budget}" + "\n" +
    r"\end{figure}"
)

# Section 4.3 (Rulo)
text = text.replace(
    r"$\theta_\mathrm{rms}=1\,\mathrm{mrad}$'da $k=2$ genlik hatası $-0{,}66\pm2{,}1\%$.",
    r"$\theta_\mathrm{rms}=1\,\mathrm{mrad}$'da $k=2$ genlik hatası $-0{,}66\pm2{,}1\%$'dir. Şekil~\ref{fig:tilt_budget}, rulo açısından kaynaklanan sistematik hata artışını göstermektedir." + "\n\n" +
    r"\begin{figure}[ht]" + "\n" +
    r"\centering" + "\n" +
    r"\includegraphics[width=0.7\textwidth]{fig7_tilt_model.png}" + "\n" +
    r"\caption{Kuadrupol rulosunun (tilt) sistematik hataya etkisi. Modlar arasındaki hata eğilimleri rulonun genliğiyle ölçeklenmektedir.}" + "\n" +
    r"\label{fig:tilt_budget}" + "\n" +
    r"\end{figure}"
)

# Section 4.4 (Gradyan)
text = text.replace(
    r"$\sigma_G=2\%$'de $\pm16\%$'ya çıkar, bu nedenle sıkı gradyan kalibrasyonu önem taşır.",
    r"$\sigma_G=2\%$'de $\pm16\%$'ya çıkar, bu nedenle sıkı gradyan kalibrasyonu önem taşır. Şekil~\ref{fig:gradient_budget}, gradyan hatasının artmasıyla gözlemlenen genlik hatasını özetlemektedir." + "\n\n" +
    r"\begin{figure}[ht]" + "\n" +
    r"\centering" + "\n" +
    r"\includegraphics[width=0.7\textwidth]{fig4_sigma_model.png}" + "\n" +
    r"\caption{Sistematik gradyan artışının $k=1,2,3$ modlarındaki göreli genlik hatasına etkisi. Tüm BPM gürültüleri sıfırlanmış olup saf gradyan kayması izole edilmiştir.}" + "\n" +
    r"\label{fig:gradient_budget}" + "\n" +
    r"\end{figure}"
)

# Fix SVD Caption
text = text.replace(
    r"\caption{Tepki matrisi SVD analizi. En büyük iki tekil değer diğerlerinden" + "\n" + r"belirgin biçimde ayrışır; karşılık gelen tekil vektörler $k=2$ FODO-antisimetrik" + "\n" + r"modlarıdır. $\kappa(R)=\sigma_1/\sigma_{48}\approx249$ koşullanma sayısıdır;" + "\n" + r"bu rakam yörünge kazancı değildir.}",
    r"\caption{Fourier modlarının yörünge kazançları ($\|RF_k\|$). $k=2$ modu dikey ayar frekansına yakınlığı sebebiyle diğerlerinden belirgin biçimde ayrışır ve rezonans kuvvetlenmesi sergiler.}"
)

# Fix text references
text = text.replace("SVD analizi,", "Yörünge kazancı analizi (Şekil~\\ref{fig:svd}),")
text = text.replace(r"$R$'nin en büyük iki tekil vektörünün tam olarak $k=2$ cos ve sin" + "\n" + r"modları olduğunu ortaya koymaktadır (Şekil~\ref{fig:svd}).", r"$k=2$ modlarının yörüngeyi maksimum seviyede kuvvetlendirdiğini ortaya koymaktadır.")

with open("makale_tr.tex", "w") as f:
    f.write(text)

