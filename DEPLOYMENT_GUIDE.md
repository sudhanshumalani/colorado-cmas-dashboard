# 🚀 Quick Deployment Guide for Streamlit Community Cloud

This guide will help you deploy your CMAS Performance Dashboard to **Streamlit Community Cloud** for FREE.

## ✅ What You Get with Streamlit Cloud (FREE)
- Completely free hosting
- Shareable public URL (e.g., `https://your-app.streamlit.app`)
- Automatic updates when you push to GitHub
- No credit card required
- SSL/HTTPS enabled
- Custom domains supported
- Generous resource limits

---

## 📋 Pre-Deployment Checklist

Before deploying, make sure you have:

- [ ] GitHub account (create one at https://github.com/signup)
- [ ] All files ready:
  - [ ] `app.py`
  - [ ] `requirements.txt`
  - [ ] `README.md`
  - [ ] `2025 CMAS Performance_ELA.xlsx`
  - [ ] `2025 CMAS Performance_Math.xlsx`
  - [ ] `.streamlit/config.toml` (optional but recommended)

---

## 🎯 Step-by-Step Deployment

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `cmas-performance-dashboard` (or your choice)
3. Description: "Interactive Colorado CMAS Performance Dashboard"
4. Choose **Public** (required for free Streamlit hosting)
5. Do NOT initialize with README (you already have one)
6. Click **"Create repository"**

### Step 2: Push Your Code to GitHub

**Option A: Using GitHub Desktop (Easiest)**
1. Download GitHub Desktop: https://desktop.github.com/
2. Open GitHub Desktop
3. File → Add Local Repository → Select your project folder
4. If prompted, click "Create a repository"
5. Click "Publish repository" button
6. Uncheck "Keep this code private"
7. Click "Publish repository"

**Option B: Using Command Line**

Open terminal/command prompt in your project folder and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: CMAS Performance Dashboard"

# Add your GitHub repository as remote
# Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub username and repo name
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/cmas-dashboard.git
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io

2. **Sign In**
   - Click **"Sign in"**
   - Click **"Continue with GitHub"**
   - Authorize Streamlit to access your GitHub

3. **Create New App**
   - Click **"New app"** (or "Create app" button)

4. **Configure Deployment**
   - **Repository**: Select `YOUR_USERNAME/cmas-performance-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom name (e.g., `colorado-cmas-dashboard`)

5. **Advanced Settings** (Optional)
   - Python version: 3.11 (recommended)
   - Click "Deploy!"

6. **Wait for Deployment**
   - Initial deployment takes 2-3 minutes
   - Watch the build logs for any errors
   - Once complete, you'll see "Your app is live!"

7. **Share Your App**
   - Your app URL: `https://YOUR-APP-NAME.streamlit.app`
   - Click the share button to get the link
   - Share with colleagues, educators, stakeholders!

---

## 🔄 Updating Your Deployed App

Your app **automatically updates** when you push changes to GitHub!

```bash
# Make changes to app.py or other files
# Then commit and push:

git add .
git commit -m "Description of your changes"
git push
```

Wait 1-2 minutes and your app will automatically redeploy with the changes!

---

## 🛠️ Troubleshooting Common Issues

### Issue 1: "Module not found" error

**Solution**: Check that all required packages are in `requirements.txt`

Current requirements.txt should have:
```
streamlit==1.32.0
pandas==2.2.1
plotly==5.20.0
numpy==1.26.4
scipy==1.12.0
openpyxl==3.1.2
```

### Issue 2: "File not found" error for Excel files

**Solution**: Verify Excel files are in GitHub repository

```bash
# Check files are tracked by git
git status

# If Excel files are missing, add them:
git add "2025 CMAS Performance_ELA.xlsx"
git add "2025 CMAS Performance_Math.xlsx"
git commit -m "Add Excel data files"
git push
```

### Issue 3: App won't start or shows blank page

**Solution**: Check Streamlit Cloud logs
1. Go to your app dashboard on Streamlit Cloud
2. Click "Manage app"
3. View logs for error messages
4. Common fixes:
   - Ensure Python version is 3.9+
   - Check all file paths are relative (not absolute)
   - Verify no syntax errors in app.py

### Issue 4: GitHub authentication issues

**Solution**:
1. Make sure repository is **Public** (free tier requirement)
2. Re-authorize Streamlit at https://share.streamlit.io
3. Check GitHub app permissions

### Issue 5: App is slow or timing out

**Solution**:
- Streamlit Cloud has resource limits
- Consider reducing data size or optimizing code
- Free tier specs: 1 CPU core, 1GB RAM
- Add caching with `@st.cache_data` (already implemented)

---

## 🎨 Customizing Your App URL

### Default URL
`https://USERNAME-REPONAME-BRANCHNAME-FILENAME-HASH.streamlit.app`

### Custom Subdomain
When deploying, you can set a custom app name:
- Example: `colorado-cmas` → `https://colorado-cmas.streamlit.app`

### Custom Domain (Advanced)
Streamlit Cloud supports custom domains (e.g., `dashboard.yourschool.org`):
1. Go to App Settings → General
2. Add custom domain
3. Update DNS records with your domain provider

---

## 📊 Managing Your Deployed App

### App Dashboard
Access at: https://share.streamlit.io/

**Available Actions:**
- ▶️ **Restart app**: Force reload
- 🔄 **Reboot app**: Full restart
- 📊 **View logs**: Debug issues
- ⚙️ **Settings**: Configure app
- 🗑️ **Delete app**: Remove deployment

### App Settings
- Change app name/URL
- Set Python version
- Add secrets (for API keys, etc.)
- Configure resource limits

### Viewing Analytics
- View app usage stats
- See visitor counts
- Monitor performance

---

## 🔐 Security Best Practices

### Do NOT commit sensitive data
- No passwords
- No API keys
- No student-level data (use aggregated data only)

### Use Streamlit Secrets for sensitive config
If you need to store secrets:

1. Create `.streamlit/secrets.toml` locally (already in .gitignore)
2. Add secrets in Streamlit Cloud dashboard
3. Access in code: `st.secrets["key_name"]`

---

## 💡 Pro Tips

1. **Enable GitHub Actions** for automatic testing before deployment

2. **Use branches** for testing new features:
   ```bash
   git checkout -b feature-testing
   # Make changes
   git push origin feature-testing
   # Deploy separate Streamlit app from this branch for testing
   ```

3. **Monitor performance** with Streamlit's built-in metrics

4. **Add a changelog** to README for version tracking

5. **Enable discussions** on GitHub for user feedback

---

## 📞 Getting Help

### Streamlit Community
- Forum: https://discuss.streamlit.io/
- Documentation: https://docs.streamlit.io/
- GitHub: https://github.com/streamlit/streamlit

### Quick Links
- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Deployment FAQ: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app

---

## ✅ Deployment Checklist

Use this checklist to ensure successful deployment:

- [ ] All files committed to GitHub
- [ ] Repository is **Public**
- [ ] `requirements.txt` includes all dependencies
- [ ] Excel files are in repository
- [ ] Tested app locally with `streamlit run app.py`
- [ ] Signed into Streamlit Cloud with GitHub
- [ ] Created new app deployment
- [ ] Selected correct repository, branch, and file
- [ ] Deployment completed successfully
- [ ] App loads without errors
- [ ] Tested all filters and features
- [ ] Shared app URL with team

---

## 🎉 You're Done!

Your CMAS Performance Dashboard is now live and accessible to anyone with the link!

**Share your app:**
```
https://YOUR-APP-NAME.streamlit.app
```

**Next steps:**
- Share with colleagues and stakeholders
- Gather feedback
- Make improvements
- Push updates (they auto-deploy!)

---

**Need help?** Check the troubleshooting section above or open an issue on GitHub.

**Happy analyzing!** 📊✨
