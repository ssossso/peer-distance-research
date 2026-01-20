const SECRET = "researchgo"; // Render의 GOOGLE_SECRET과 정확히 동일해야 함

function json(output) {
  return ContentService
    .createTextOutput(JSON.stringify(output))
    .setMimeType(ContentService.MimeType.JSON);
}

// sha256 해시 앞 8자리만 (원문 노출 없음)
function sha256_8(s) {
  const str = (s === undefined || s === null) ? "" : String(s);
  const bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, str);
  return bytes
    .map(b => ("0" + (b & 0xff).toString(16)).slice(-2))
    .join("")
    .slice(0, 8);
}

function doPost(e) {
  const body = (e && e.postData && e.postData.contents) ? e.postData.contents : "{}";
  const data = JSON.parse(body || "{}");

  const received = (data.secret === undefined || data.secret === null) ? "" : String(data.secret);
  const expected = String(SECRET);

  // 비밀키 검사: 불일치 시 길이/해시 반환
  if (!received || received !== expected) {
    return json({
      status: "blocked",
      received_len: received.length,
      received_sha256_8: sha256_8(received),
      expected_len: expected.length,
      expected_sha256_8: sha256_8(expected)
    });
  }

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const action = data.action;

  if (action === "teacher_signup") {
    const sheet = ss.getSheetByName("Teachers");
    if (!sheet) return json({ status: "error", message: "Teachers 시트가 없습니다." });

    const username = (data.username || "").trim();
    const pw_hash = data.pw_hash || "";
    if (!username || !pw_hash) return json({ status: "error", message: "username/pw_hash 필요" });

    const values = sheet.getDataRange().getValues();
    for (let i = 1; i < values.length; i++) {
      if (String(values[i][0]).trim() === username) return json({ status: "exists" });
    }

    sheet.appendRow([username, pw_hash, new Date()]);
    return json({ status: "ok" });
  }

  if (action === "teacher_get") {
    const sheet = ss.getSheetByName("Teachers");
    if (!sheet) return json({ status: "error", message: "Teachers 시트가 없습니다." });

    const username = (data.username || "").trim();
    if (!username) return json({ status: "error", message: "username 필요" });

    const values = sheet.getDataRange().getValues();
    for (let i = 1; i < values.length; i++) {
      if (String(values[i][0]).trim() === username) {
        return json({ status: "ok", pw_hash: values[i][1] });
      }
    }
    return json({ status: "not_found" });
  }

  if (action === "result_append") {
    const sheet = ss.getSheetByName("Results");
    if (!sheet) return json({ status: "error", message: "Results 시트가 없습니다." });

    sheet.appendRow([
      data.teacher || "",
      data.class_code || "",
      data.student || "",
      data.session || "",
      JSON.stringify(data.placements || {}),
      new Date(),
      data.ip || ""
    ]);
    return json({ status: "ok" });
  }

  return json({ status: "error", message: "unknown action" });
}
