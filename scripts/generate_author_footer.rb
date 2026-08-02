#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "cgi"

config_path = File.expand_path("../includes/author-footer.yml", __dir__)
output_path = File.expand_path("../includes/_author-footer.generated.html", __dir__)

config = YAML.load_file(config_path)

def md_links_to_html(text)
  CGI.escapeHTML(text.to_s.strip.gsub(/\s+/, " ")).gsub(
    /\[([^\]]+)\]\(([^)]+)\)/,
    '<a href="\2">\1</a>'
  )
end

name = CGI.escapeHTML(config.fetch("name"))
image = CGI.escapeHTML(config.fetch("image"))
image_alt = CGI.escapeHTML(config["image-alt"] || config.fetch("name"))
label = CGI.escapeHTML(config["label"] || "About the author")
bio = md_links_to_html(config.fetch("bio"))
more_text = CGI.escapeHTML(config["more-text"] || "Read more →")
more_href = CGI.escapeHTML(config["more-href"] || "/about.html")
script = CGI.escapeHTML(config["script"] || "/assets/js/blog-post-nav.js")

html = <<~HTML
  <nav class="blog-post-nav" id="blog-post-nav" aria-label="Post navigation" hidden>
    <div class="blog-post-nav-prev" id="blog-post-nav-prev" hidden>
      <div class="blog-post-nav-label">← Previous</div>
      <a class="blog-post-nav-link" href="#"></a>
    </div>
    <div class="blog-post-nav-next" id="blog-post-nav-next" hidden>
      <div class="blog-post-nav-label">Next →</div>
      <a class="blog-post-nav-link" href="#"></a>
    </div>
  </nav>

  <section class="about-the-author" aria-label="#{label}">
    <img
      class="about-the-author-photo"
      src="#{image}"
      alt="#{image_alt}"
      width="140"
      height="140"
      loading="lazy"
    >
    <div class="about-the-author-label">#{label}</div>
    <h3 class="about-the-author-name">#{name}</h3>
    <p class="about-the-author-bio">
      #{bio}
      <a class="about-the-author-more" href="#{more_href}">#{more_text}</a>
    </p>
  </section>

  <script src="#{script}"></script>
HTML

File.write(output_path, html)
puts "Wrote #{output_path}"
