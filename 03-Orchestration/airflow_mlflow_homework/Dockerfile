FROM astrocrpublic.azurecr.io/runtime:3.0-2

# Switch to root to copy files and set permissions
USER root

# Create credentials directory and copy service account key
RUN mkdir -p /home/astro/credentials
COPY --chown=astro:astro credentials/serviceaccount.json /home/astro/credentials/serviceaccount.json
RUN chmod 600 /home/astro/credentials/serviceaccount.json

# Set environment variable for GCP authentication
ENV GOOGLE_APPLICATION_CREDENTIALS=/home/astro/credentials/serviceaccount.json
ENV GIT_PYTHON_REFRESH=quiet

# Switch back to astro user
USER astro