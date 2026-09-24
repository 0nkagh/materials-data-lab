# GitHub CLI (gh) Bulunamadı veya Oturum Açılmamış

Sisteminizde `gh` (GitHub CLI) komutu bulunamadı. Lütfen aşağıdaki adımları takip ederek kurulumu ve yetkilendirmeyi gerçekleştirin:

1. Terminali açın ve şu komutla GitHub CLI aracını kurun:
   ```bash
   winget install GitHub.cli
   ```

2. Kurulum tamamlandıktan sonra yeni bir terminal açarak (veya mevcut terminali yeniden başlatarak) GitHub hesabınıza giriş yapın:
   ```bash
   gh auth login
   ```
   (Etkileşimli giriş işlemi sadece sizin tarafınızdan yapılmalıdır).

3. Yukarıdaki adımları başarıyla tamamladıktan sonra, bir önceki promptu **yeniden gönderin**.
